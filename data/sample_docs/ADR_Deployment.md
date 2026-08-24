# ADR-004: Deploy OCR Workloads on Kubernetes

## Decision
The Government OCR Platform will deploy the OCR Service and API Gateway on Kubernetes. Document Storage and the Monitoring Service remain managed platform dependencies accessed through private service endpoints.

## Context
The platform needs repeatable deployments, horizontal scaling for intake peaks, and controlled promotion from the UAT Environment to production. Kubernetes is already operated by the Platform Engineering Team.

## Consequences
Kubernetes improves deployment consistency and allows OCR Service workers to scale independently. It also introduces capacity, networking, and patching responsibilities for Platform Engineering Team. Monitoring Service dashboards must cover pod health, queue depth, API latency, and failed OCR jobs.

## Security Review
The Security Team must approve workload identities, network policies, and secret management before production deployment. The API Gateway remains the controlled entry point for all external clients.

## Related Risks
Kubernetes capacity is tracked as R-001 in the Risk Register. Batch processing failure is covered by the incident runbook and validated in the UAT Environment.
