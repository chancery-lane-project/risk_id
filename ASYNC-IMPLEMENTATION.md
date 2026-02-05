# Async Task Processing Implementation

## Status: READY TO IMPLEMENT

## What We've Created:

### 1. Task Manager (`tclp/task_manager.py`) ✅
- SQLite database for task persistence
- Functions: `create_task()`, `update_task()`, `get_task()`, `cleanup_old_tasks()`
- Stores: task_id, status, progress, result, error

### 2. Background Tasks (`tclp/background_tasks.py`) ✅
- `process_contract_task()` - Runs ML processing in background
- Updates progress: 10% → 20% → 30% → ... → 100%
- Stores result in database when complete

### 3. What Still Needs to Be Done:

#### A. Update `/process/` endpoint in `tclp/app.py`
Replace lines 101-240 with:
```python
@app.post("/process/")
async def process_contract(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Endpoint to process a contract file.
    Returns immediately with a task_id for polling.
    """
    # Check validity
    allowed_extensions = ['.txt', '.pdf', '.docx', '.doc', '.md']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return JSONResponse(
            content={
                "error": f"Only {', '.join(allowed_extensions)} files are supported."
            },
            status_code=400,
        )

    # Read file content
    file_content = await file.read()

    # Create task
    task_id = task_manager.create_task()
    print(f"[INFO] Created task {task_id} for file: {file.filename}")

    # Start background processing
    import background_tasks as bg_tasks
    background_tasks.add_task(
        bg_tasks.process_contract_task,
        task_id=task_id,
        file_content=file_content,
        filename=file.filename,
        temp_dir=temp_dir,
        output_dir=output_dir,
        tokenizer=tokenizer,
        d_model=d_model,
        c_model=c_model,
        embedding_cache=embedding_cache,
        CAT0=CAT0,
        CAT1=CAT1,
        CAT2=CAT2,
        CAT3=CAT3
    )

    return {"task_id": task_id, "status": "processing"}
```

#### B. Add `/task/{task_id}` endpoint in `tclp/app.py`
Insert after the new `/process/` endpoint:
```python
@app.get("/task/{task_id}")
def get_task_status(task_id: str):
    """
    Get the status of a background task.
    Returns task status, progress, and result (if completed).
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Return task info
    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
        "created_at": task["created_at"],
        "updated_at": task["updated_at"]
    }

    if task["status"] == "completed":
        response["result"] = task["result"]
    elif task["status"] == "failed":
        response["error"] = task["error"]

    return response
```

#### C. Update Frontend (`tclp/provocotype-1/index.htm`)
Replace the current fetch logic (around line 315) with polling:
```javascript
// Submit and get task ID
const resp = await fetch('process/', { method: 'POST', body: formData });
const { task_id } = await resp.json();

// Poll for completion
const pollTask = async () => {
  const statusResp = await fetch(`task/${task_id}`);
  const taskStatus = await statusResp.json();

  if (taskStatus.status === 'completed') {
    // Use taskStatus.result (same structure as before)
    const data = taskStatus.result;
    // ... rest of existing code ...
  } else if (taskStatus.status === 'failed') {
    throw new Error(taskStatus.error);
  } else {
    // Still processing, poll again in 2 seconds
    setTimeout(pollTask, 2000);
  }
};

await pollTask();
```

## Benefits:
- ✅ No more timeouts (returns immediately)
- ✅ Progress tracking (10% → 100%)
- ✅ Task persistence (survives restarts)
- ✅ Single server (no Redis needed)
- ✅ Better error handling

## Testing Plan:
1. Deploy changes
2. Upload a file
3. Check browser console for task_id
4. Watch server logs for progress updates
5. Verify results page loads correctly

## Ready to proceed?
User confirmed - implementing now!
