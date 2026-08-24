# Incident: Batch OCR Failure

## Summary
On 2025-04-18, a scheduled batch of 12,400 documents experienced elevated failures in the OCR Service. The API Gateway remained available, but Kubernetes worker pods reached their memory limit and jobs stayed pending.

## Impact
The Operations Team reported delayed results for the UAT Environment and a six-hour processing backlog. No source documents were lost because Document Storage remained available.

## Detection
The Monitoring Service detected increased pending jobs and worker restarts. The alert did not include a direct escalation to Platform Engineering Team, which slowed diagnosis.

## Resolution
Platform Engineering Team increased Kubernetes worker capacity, restarted affected OCR Service workers, and replayed pending jobs using the existing job identifiers. The Operations Team verified completion through the API status endpoint.

## Follow-up Actions
Add queue-depth alerts, test batch limits in the UAT Environment, and document capacity thresholds in the Deployment Guide. The event is linked to risk R-001 in the Risk Register.
