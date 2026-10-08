"""Conservative lexical procedure support, not general semantic entailment."""
import re
from backend.observability import emit

_OPS = {'deploy': ('deploy', 'deployment'), 'install': ('install', 'installation'),
        'configure': ('configure', 'configuration'), 'restore': ('restore', 'restoration'),
        'restart': ('restart',), 'migrate': ('migrate', 'migration'),
        'rollback': ('rollback', 'roll back'), 'rotate': ('rotate', 'rotation')}
_ACTIONS = r'(?:deploy|install|configure|restore|restart|migrate|rotate|verify|validate|run|stop|start|apply|create|copy|execute|connect|enable|disable|roll back)'
_EXCLUDE = re.compile(r'prerequisites?|pre[- ]deployment|before\s+(?:deployment|deploying)|incident|remediation|follow[- ]up|preconditions?', re.I)


def requested_operations(question):
    q = question.casefold()
    if not re.search(r'\bsteps?\s+(?:to|for)|\bprocedure\s+(?:for|to)|\bhow\s+to\b|\b\w+\s+sequence\b', q):
        return []
    trigger = re.search(r'\bsteps?\s+(?:to|for)|\bprocedure\s+(?:for|to)|\bhow\s+to\b', q)
    if trigger:
        q = q[trigger.start():]
    operations = []
    for clause in re.split(r'\band\b', q):
        previous_count = len(operations)
        for operation, words in _OPS.items():
            if any(re.search(r'\b' + word + r'\b', clause) for word in words):
                # Ignore a source reference such as "according to the deployment guide".
                if re.search(r'\bsteps?\s+(?:to|for)|\bprocedure\s+(?:for|to)|\bhow\s+to\b|\bsequence\b', clause) or operations:
                    operations.append((operation, set(re.findall(r'\b[a-z]+\b', clause)) & {'archive', 'database', 'credentials', 'certificate', 'worker', 'backup'}))
        # A procedural conjunction must not silently discard an unsupported
        # request. Conservatively refuse unknown clauses, including when an
        # earlier clause has a recognized operation.
        if len(operations) == previous_count:
            operations.append(('unknown', set()))
    # Explicit procedures outside the supported vocabulary fail closed.
    return operations or [('unknown', set())]


def procedure_evidence(question, evidence):
    """Return only chunks supporting every requested operation, else None."""
    operations = requested_operations(question)
    if not operations:
        return []
    used = []
    for operation, anchors in operations:
        found = None
        for item in evidence:
            context = item.section + ' ' + item.text
            if _EXCLUDE.search(context):
                continue
            if not any(re.search(r'\b' + word + r'\b', context, re.I) for word in _OPS.get(operation, ())):
                continue
            if not anchors <= set(re.findall(r'\b[a-z]+\b', context.casefold())):
                continue
            numbered = re.findall(r'^\s*\d+[.)]\s+(.+)$', item.text, re.M)
            sequence = bool(re.search(r'\bfirst\b.*\bthen\b', item.text, re.I | re.S))
            actions = re.findall(r'(?im)(?:^\s*\d+[.)]\s*|^|[.;]\s*|\bthen\s+)'+_ACTIONS+r'\b', item.text)
            primary_action = bool(re.search(r'(?im)(?:^\s*\d+[.)]\s*|^|[.;]\s*|\bthen\s+)' + re.escape(operation) + r'\b', item.text))
            if primary_action and len(actions) >= 2 and (len(numbered) >= 2 or sequence):
                found = item
                break
        if found is None:
            emit('procedural_support_check', supported=False, reason_code='missing_procedural_support')
            return None
        if found not in used:
            used.append(found)
    emit('procedural_support_check', supported=True, reason_code='procedural_support_present')
    return used
