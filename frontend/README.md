# Enterprise Knowledge Copilot frontend

React + TypeScript + Vite, Tailwind CSS. Live API answers only; fixtures are confined to tests. Backend logic, retrieval and deployment configuration are unchanged.

## Local preview

Requires Node 22.12+ (validated with Node 24) and npm.

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Open http://localhost:5173. `VITE_API_BASE_URL=/api` uses the local Vite proxy to the verified Render API. To use a local backend, set `API_PROXY_TARGET=http://127.0.0.1:8000`. Environment changes require restarting Vite. No keys or credentials belong in VITE variables: these are public build-time settings.

```powershell
npm test
npm run build
npm run preview
```

`preview` serves the built UI only and does not supply the development API proxy. For interactive API testing use `npm run dev`, or configure a same-origin hosting reverse proxy before testing the built site.

## Browser/API deployment boundary

The deployed API currently returns 405 for CORS preflight and has no Allow-Origin header. Direct cross-origin browser requests will fail. Local preview works through the Vite development proxy; this does not change backend security. Production needs either a same-origin `/api` reverse proxy or an explicitly approved backend CORS change allowing only the actual frontend HTTPS origin, POST/OPTIONS and Content-Type, exposing X-Request-ID and Retry-After. No wildcard origin and no credentials are needed. Configure `VITE_API_BASE_URL` to that proxy path or the API URL after exact-origin CORS is approved. Do not deploy the development server publicly.

## Limits and privacy

Questions max 2,000 characters; session IDs are UUIDs and remain in memory only. The successful server session ID is reused sequentially; a new conversation resets it. Local idle expiration is 30 minutes, matching backend TTL. The backend can evict sessions earlier (256 capacity) or lose them on restart. It retains at most five turns. Conversation history is not persisted in browser storage. Questions are sent to the configured API; use only synthetic demo content.

A request times out at 90 seconds, with cold-start guidance after eight seconds. No automatic retry or background polling adds load. Errors 422/429/500/503 are shown safely, with a request ID where available; network errors include proxy/CORS guidance. Render Free may sleep and rebuild its demo corpus after replacement; this UI does not change persistent-volume compatibility.

The grounding badge reflects the backend decision, not an independent entailment guarantee. Citations and retrieved provenance are separated; refusal evidence is explicitly not claimed as supporting citations. Scores are raw API retrieval signals, not confidence. Source text is displayed as escaped text, never HTML. No document download endpoint, uploads, authentication or streaming is added.

## Verification

`npm test` tests integration, citations, refusal, sessions, loading, escaped output, response validation, and error statuses using isolated fixtures. `npm run build` runs strict TypeScript checks and production bundling. These tests are not live backend validation. Frontend CI integration and hosting are deferred; the existing Python/Docker CI remains unchanged.

## Windows reinstall troubleshooting

Stop any running Vite preview with Ctrl+C in its terminal before `npm ci`. Windows locks loaded native modules (including Lightning CSS); reinstalling while Vite runs can fail with `EPERM unlink` and leave dependencies partially removed. Once the preview is stopped, run `npm ci` again using the existing lockfile. Then run `npm run build`, `npm test`, and `npm run dev`. Do not delete the lockfile or install Vite globally. Paste only commands from code blocks into PowerShell.

## Conversation workspace (Phase 5B)

The sidebar lists real conversations created in this tab, using the first submitted question as the title. Switching restores that conversation's answers, draft and server session ID; New conversation starts a fresh session while retaining earlier local conversations. Up to 20 conversations are kept in memory, with older entries removed when the limit is exceeded. Reloading the page clears the list. This is not persistent server-side history. Switching and reset are disabled during a request so responses cannot land in the wrong conversation. Backend expiry, eviction and restarts can still remove follow-up context even while local answers remain visible.

The compact workspace keeps the composer near the top and removes the introductory hero after a query. Answers, grounding decisions, citations and expandable evidence remain the primary content. On mobile the conversation list scrolls horizontally. Browser visual review is still required for desktop and mobile; run `npm run dev` and inspect both populated and refusal states. Stop Vite before reinstalling dependencies on Windows.

## Answer Inspector (Phase 5E)

The three-column desktop layout follows the latest turn in the active conversation; smaller screens stack the inspector below the workspace. Knowledge Graph displays only typed graph pairs returned by the API, Sources shows real cited documents and provenance, and Verification separates evidence sufficiency from unimplemented claim verification, numeric/date consistency and human approval. See [GROUNDING_DESIGN.md](./GROUNDING_DESIGN.md) for the audited inventory and future backend proposal. No backend behavior changes.

## Final sprint release configuration

Backend `/query` adds `verification` and `review` without removing existing fields. SUPPORTED means literal cited-source agreement, not general entailment. NOT_VERIFIED is deliberately retained for unsupported rule coverage. Critical numeric/date claims which cannot be confirmed refuse safely. Operational tool text remains not verified against document citations. Review is heuristic signaling only; there is no queue, approval or authenticated reviewer.

`GET /graph` exposes at most 80 nodes and 160 edges from bundled synthetic documents only, excluding uploaded knowledge. The corpus view is independent of answer-specific provenance.

For Cloudflare Pages use root `frontend`, build `npm ci && npm run build`, output `dist`, Node 24, and `VITE_API_BASE_URL=https://enterprise-knowledge-copilot-0e7q.onrender.com`. In Render set `FRONTEND_ORIGINS` to the actual exact Pages HTTPS origin (no trailing slash, path, wildcard or credentials), then redeploy the verified API commit. The actual Pages project hostname must be confirmed first; preview branch origins are not automatically allowed. Vite env variables are public; no API keys belong there. Local `/api` development proxy continues to work.

Deployment is blocked until CI's exact image build and container checks pass, including ephemeral reconstruction and persistent-volume validation. Render Free remains ephemeral and can cold-start; sessions are process-local. No service settings are changed by this code. Test graph access, query, refusal, CORS rejection of another origin, and mobile inspector after deployment. Keep the prior verified commit for rollback.
