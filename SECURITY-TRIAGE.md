# Security Vulnerability Triage

## 🔴 CRITICAL - Fix Immediately (2 issues)

### 1. PyTorch RCE via `torch.load`
**Package:** `torch` (currently pinned to 2.1.2)
**Issue:** Remote code execution when loading models
**Impact:** HIGH - Core ML functionality
**Effort:** MEDIUM - Need to test model compatibility
**Action:** Update to torch >= 2.5.0
```bash
poetry update torch
# Test model loading after update
```

### 2. h11 Malformed Chunked-Encoding
**Package:** `h11` (0.14.0)
**Issue:** Accepts malformed chunked encoding bodies
**Impact:** MEDIUM - Used by HTTP stack
**Effort:** LOW - Likely safe to update
**Action:** Update to h11 >= 0.14.1
```bash
poetry update h11
```

---

## 🟠 HIGH PRIORITY - Fix Before Public Launch (21 issues)

### Group 1: HTTP/Network Stack (HIGHEST PRIORITY)
These affect your web server and could be exploited remotely.

#### urllib3 (3 issues) - Decompression vulnerabilities
**Current:** 2.2.3
**Impact:** HIGH - DoS attacks possible
**Effort:** LOW - Drop-in replacement
**Action:** Update to latest urllib3
```bash
poetry update urllib3
```

#### python-multipart (2 issues) - DoS & Arbitrary File Write
**Current:** 0.0.17
**Impact:** HIGH - File upload handling
**Effort:** LOW
**Action:** Update to latest python-multipart
```bash
poetry update python-multipart
```

#### starlette - O(n^2) DoS
**Current:** 0.41.3
**Impact:** HIGH - Core FastAPI dependency
**Effort:** LOW - Should be compatible
**Action:** Update starlette
```bash
poetry update starlette
```

**Testing Required:** Test file upload endpoints after these updates

---

### Group 2: ML Model Loading (MEDIUM PRIORITY)
These affect model/data loading but you control the models.

#### transformers (3 High + many Moderate ReDoS)
**Current:** 4.46.2
**Issue:** Deserialization of untrusted data, ReDoS attacks
**Impact:** MEDIUM - You control model sources
**Effort:** MEDIUM - Test model loading
**Action:** Update transformers
```bash
poetry update transformers
```

#### torch (additional issues)
- Heap buffer overflow
- Use-after-free
- Improper resource shutdown

**Action:** Already covered in Critical section

---

### Group 3: Development Dependencies (LOWER PRIORITY FOR PRODUCTION)
These mainly affect development/notebook environments.

#### notebook, nbconvert, jupyterlab
**Impact:** LOW for production (only if running notebooks in prod)
**Effort:** LOW
**Decision:** Update if you use notebooks in production, otherwise safe to ignore
```bash
poetry update notebook nbconvert jupyterlab
```

---

### Group 4: Other High-Risk Dependencies

#### setuptools - Path Traversal
**Current:** 75.1.0
**Impact:** LOW - Build-time only
**Effort:** LOW
**Action:** Update setuptools
```bash
poetry update setuptools
```

#### wheel - Arbitrary File Permission
**Current:** 0.44.0
**Impact:** LOW - Install-time only
**Effort:** LOW
**Action:** Update wheel
```bash
poetry update wheel
```

#### aiohttp - Zip Bomb
**Current:** Not directly used (transitive dependency)
**Impact:** MEDIUM
**Effort:** LOW
**Action:** Update parent packages
```bash
poetry update
```

---

## 🟡 MODERATE PRIORITY - Address Post-Launch (30 issues)

### Jinja2 Sandbox Breakouts (3 issues)
**Impact:** LOW - Not using Jinja2 sandbox features
**Effort:** LOW
**Action:** Update jinja2
```bash
poetry update jinja2
```

### Transformers ReDoS (10+ issues)
**Impact:** LOW - ReDoS requires malicious input to specific functions
**Effort:** LOW - Covered by Group 2 transformers update
**Action:** Same as above

### aiohttp Multiple Issues (5+ issues)
**Impact:** LOW - Transitive dependency
**Effort:** LOW
**Action:** Update dependencies
```bash
poetry update aiohttp
```

### Others
- tornado (2 DoS issues) - Update if needed
- requests (.netrc leak) - Update
- protobuf (JSON/DoS) - Update

---

## 🟢 LOW PRIORITY - Monitor (10 issues)

These are informational or have minimal impact:
- AIOHTTP path leaks, cookie warnings
- PyTorch local DoS (requires local access)
- pip path traversal (build-time)
- cryptography OpenSSL (informational)

**Action:** Address during regular maintenance cycles

---

## 📋 RECOMMENDED ACTION PLAN

### Phase 1: Pre-Launch (Do Now)
**Time Estimate: 1-2 hours**

```bash
# 1. Update critical packages
poetry update torch h11

# 2. Update high-priority HTTP stack
poetry update urllib3 python-multipart starlette

# 3. Test the application
cd tclp && poetry run python -c "from app import app; print('✓ Import successful')"
poetry run uvicorn tclp.app:app --host 127.0.0.1 --port 8000 &
# Test file upload, API endpoints, model loading
curl http://127.0.0.1:8000/
# Kill test server

# 4. Update ML dependencies (test carefully)
poetry update transformers

# 5. Commit and test thoroughly
git add poetry.lock
git commit -m "security: update critical dependencies for public deployment"
```

### Phase 2: Post-Launch (Within 1 week)
**Time Estimate: 30 minutes**

```bash
# Update remaining high-priority items
poetry update setuptools wheel jinja2

# Update development dependencies if used
poetry update --group dev

# Commit
git add poetry.lock
git commit -m "security: update moderate-priority dependencies"
```

### Phase 3: Regular Maintenance (Monthly)
**Time Estimate: 15 minutes**

```bash
# Full dependency update
poetry update

# Review and test
# Commit if tests pass
```

---

## ⚠️ TESTING CHECKLIST

After updates, verify:

- [ ] Application starts without errors
- [ ] Models load correctly (CC_BERT, clustering, UMAP)
- [ ] File upload works (POST /process/)
- [ ] Clause recommendations work (POST /find_clauses/)
- [ ] Static files serve correctly
- [ ] Frontend displays properly
- [ ] No new warnings in logs

---

## 🎯 SUMMARY

**Critical Issues:** 2 (torch RCE, h11 chunked encoding)
**High Priority for Public Hosting:** 8 packages (HTTP stack, file uploads)
**Estimated Time to Fix Critical + High:** 1-2 hours
**Risk if Not Fixed:** RCE, DoS attacks, file system access

**Recommendation:** Run Phase 1 updates before public deployment. The effort is manageable and significantly reduces attack surface.
