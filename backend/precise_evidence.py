"""Bounded, source-extractive support for procedural, causal, flow and team queries."""
import re
from backend.procedural_support import requested_operations,procedure_evidence

IGNORE={'what','which','why','how','does','the','are','was','were','did','a','an','of','to','and','for','from','between','during','root','cause','caused','incident','failure','fail','processing','left'}
def kind(question):
 q=question.casefold()
 if requested_operations(question):return 'procedure'
 if re.search(r'\b(?:cause|caused|why)\b',q) and re.search(r'\b(?:incident|fail|failure|jobs|batch)\b',q):return 'cause'
 if re.search(r'\b(?:interact|routed|routes|sends|flow)\b',q) and not re.search(r'\b(?:protocol|price|cost|protects|date)\b',q):return 'flow'
 if not re.search(r'\b(?:risk|impact|notify|notified)\b',q) and (re.search(r'\b(?:teams|responsibilities)\b',q) or ('operations' in q and 'security' in q)) and re.search(r'\b(?:own|owns|operations|security|maintain|handles|responsibilities)\b',q):return 'ownership'
 return ''

def select(question,evidence):
 intent=kind(question)
 if intent=='procedure':
  used=procedure_evidence(question,evidence)
  return [(e.text,e) for e in used] if used else []
 anchors=set(re.findall(r'[a-z]+',question.casefold()))-IGNORE
 candidates=[]
 for e in evidence:
  sentences=re.split(r'(?<=[.!?])\s+',re.sub(r'\s+',' ',e.text).strip())
  lines=[]
  for text in sentences:
   low=text.casefold();overlap=anchors & set(re.findall(r'[a-z]+',low))
   context_overlap=anchors & set(re.findall(r'[a-z]+',e.text.casefold()))
   if intent=='cause' and context_overlap and all(set(re.findall(r'[a-z]+',name.casefold())) <= set(re.findall(r'[a-z]+',e.text.casefold())) for name in re.findall(r'\b[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)+',question)) and re.search(r'\b(?:because|due to|exhausted|exceeded|reached)\b',low) and not re.search(r'\b(?:mitigation|risk|could|may)\b',low):lines.append(text)
   if intent=='flow' and len(overlap)>=2 and re.search(r'\b(?:submits|validates|sends|routes|stores|publishes)\b',low):lines.append(text)
   if intent=='ownership' and overlap and re.search(r'\bteam\b.*?\b(?:owns|reviews|patches|maintains|responds|approves)\b',low):lines.append(text)
  if lines:candidates.append((len(lines),e,lines))
 candidates.sort(key=lambda row:(-row[0],row[1].source_document,row[1].chunk_id))
 if intent in {'cause','flow'}:candidates=candidates[:1]
 elif intent=='ownership':candidates=candidates[:2]
 else:return []
 return [(text,e) for _,e,lines in candidates for text in lines]
