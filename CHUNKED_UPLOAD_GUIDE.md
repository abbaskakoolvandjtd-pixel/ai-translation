# Chunked File Upload Guide

## Overview
The translation service now supports chunked file uploads, allowing you to upload PDF files larger than the server's single-request limit (e.g., 25MB). Files up to 1GB can be uploaded by splitting them into smaller chunks.

## Configuration
Edit `.env` or set environment variables:
- `MAX_UPLOAD_SIZE_MB`: Maximum total file size (default: 1000MB = 1GB)
- `CHUNK_SIZE_MB`: Size of each chunk (default: 5MB)

## API Endpoints

### 1. Direct Upload (for small files < 5MB)
```
POST /translation/v1/upload
Content-Type: multipart/form-data

file: <pdf_file>
```

### 2. Initiate Chunked Upload
```
POST /translation/v1/upload/initiate
Content-Type: application/x-www-form-urlencoded

filename: document.pdf
total_size: 52428800
```

Response:
```json
{
  "upload_session_id": "abc123def456",
  "session_id": "uuid-here",
  "filename": "document.pdf",
  "total_size": 52428800,
  "chunk_size": 5242880,
  "num_chunks": 10,
  "message": "Session initialized. Upload 10 chunks sequentially."
}
```

### 3. Upload Each Chunk
```
POST /translation/v1/upload/chunk
Content-Type: multipart/form-data

upload_session_id: abc123def456
chunk_index: 0
chunk_data: <chunk_file>
```

Repeat for each chunk (0 to num_chunks-1).

Response:
```json
{
  "upload_session_id": "abc123def456",
  "chunk_index": 0,
  "chunk_size": 5242880,
  "status": "uploaded",
  "all_chunks_uploaded": false,
  "message": "Chunk 1/10 uploaded successfully"
}
```

### 4. Complete Upload
```
POST /translation/v1/upload/complete
Content-Type: application/x-www-form-urlencoded

upload_session_id: abc123def456
```

Response:
```json
{
  "job_id": "job-uuid-here",
  "status": "PENDING",
  "message": "File assembled and uploaded (50.00MB). Worker will pick it up shortly."
}
```

## Client-Side Implementation Example (JavaScript)

```javascript
const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB

async function uploadLargeFile(file, authToken) {
  const filename = file.name;
  const totalSize = file.size;
  
  // Step 1: Initiate session
  const initiateForm = new FormData();
  initiateForm.append('filename', filename);
  initiateForm.append('total_size', totalSize.toString());
  
  const initiateResp = await fetch('/translation/v1/upload/initiate', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${authToken}` },
    body: initiateForm
  });
  const session = await initiateResp.json();
  
  // Step 2: Upload chunks
  for (let i = 0; i < session.num_chunks; i++) {
    const start = i * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, totalSize);
    const chunk = file.slice(start, end);
    
    const chunkForm = new FormData();
    chunkForm.append('upload_session_id', session.upload_session_id);
    chunkForm.append('chunk_index', i.toString());
    chunkForm.append('chunk_data', chunk, `chunk_${i}`);
    
    const chunkResp = await fetch('/translation/v1/upload/chunk', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${authToken}` },
      body: chunkForm
    });
    
    console.log(`Uploaded chunk ${i + 1}/${session.num_chunks}`);
  }
  
  // Step 3: Complete upload
  const completeForm = new FormData();
  completeForm.append('upload_session_id', session.upload_session_id);
  
  const completeResp = await fetch('/translation/v1/upload/complete', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${authToken}` },
    body: completeForm
  });
  
  const result = await completeResp.json();
  console.log('Upload complete! Job ID:', result.job_id);
  return result.job_id;
}
```

## Error Handling
- **413 Payload Too Large**: File exceeds maximum size
- **404 Not Found**: Invalid session ID
- **400 Bad Request**: Invalid chunk index or size mismatch
- **403 Forbidden**: User lacks permission or session belongs to another user

## Benefits
- Bypasses server request size limits
- Resumable uploads (can retry failed chunks)
- Progress tracking
- Secure session-based authentication
