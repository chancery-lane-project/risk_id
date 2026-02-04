# Frontend API Compatibility with BASE_PATH

## Analysis

The frontend in `tclp/provocotype-1/index.htm` uses **relative paths** for all API calls:
- `/process/` (line 315)
- `/find_clauses/` (line 327)

## FastAPI root_path Behavior

FastAPI's `root_path` parameter (set in Task 2) automatically handles BASE_PATH prefixing:

1. **Route Decorators**: All routes defined with `@app.get()`, `@app.post()`, etc. are automatically prefixed
   - Route defined as `@app.post("/process/")` becomes accessible at `{BASE_PATH}/process/`
   - Example: With `BASE_PATH=/risk-id`, the route is available at `/risk-id/process/`

2. **Static File Mounts**: Already manually prefixed in Task 2
   - `app.mount(f"{BASE_PATH}/assets", ...)` 
   - `app.mount(f"{BASE_PATH}/output", ...)`

3. **Frontend Requests**: Relative paths work transparently
   - When frontend is served from `/risk-id/`, the fetch to `/process/` resolves to `/risk-id/process/`
   - Browser automatically prepends the current path context

## Verification

✅ **No frontend code changes needed**
✅ **FastAPI handles routing automatically via root_path**
✅ **Static assets properly prefixed in Task 2**

## Testing

Test with BASE_PATH set:
```bash
BASE_PATH=/risk-id poetry run uvicorn tclp.app:app --host 127.0.0.1 --port 8000
```

Access at: http://127.0.0.1:8000/risk-id/
