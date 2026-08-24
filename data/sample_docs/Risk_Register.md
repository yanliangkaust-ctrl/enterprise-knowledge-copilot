# Government OCR Platform Risk Register

## Active Risks

### R-001: Kubernetes Capacity During Batch Peaks
- **Impact:** OCR jobs remain pending and processing commitments may be missed.
- **Owner:** Platform Engineering Team
- **Mitigation:** Capacity thresholds, queue-depth alerts in the Monitoring Service, and batch testing in the UAT Environment.
- **Related evidence:** ADR-004 and the Batch OCR Failure incident.

### R-002: Unauthorized Document Access
- **Impact:** Sensitive public-service documents could be exposed.
- **Owner:** Security Team
- **Mitigation:** Least-privilege identities, TLS at the API Gateway, encrypted Document Storage, and audit review.
- **Related evidence:** Security Requirements and API Specification.

### R-003: Incomplete Operational Alerts
- **Impact:** Operations Team may detect failed OCR jobs late.
- **Owner:** Operations Team
- **Mitigation:** Monitoring Service coverage for worker restarts, queue depth, latency, and failed jobs.
- **Related evidence:** Batch OCR Failure incident.

## Review Cadence
The Security Team and Platform Engineering Team review risks before each release from the UAT Environment to production.
