# Tasks: Asynchronous Multi-Topic Research Implementation

A checklist for converting `research_topic` to support asynchronous operation with multi-topic research, caching, and concurrency control.

## Success Criteria

- ✅ `research_topic` returns immediately with job ID
- ✅ Support for 1-10 topics per research job  
- ✅ Intelligent concurrent processing with rate limiting
- ✅ Complete job status tracking and management
- ✅ Resource usage within acceptable bounds
- ✅ Backward compatibility maintained
- ✅ Comprehensive test coverage (>85%)
- ✅ Updated documentation with clear examples

## Implementation Checklist

- [x] **Task 1.1**: Create job management data models (`src/ragster/job_models.py`)
  - JobStatus enum, TopicProgress dataclass, ResearchJob dataclass, JobMetrics dataclass
- [x] **Task 1.2**: Implement JobManager class (`src/ragster/job_manager.py`)
  - CRUD operations, thread-safe storage, state transitions, task tracking
- [x] **Task 1.3**: Update LoadTopicToolArgs for multi-topic support
  - Change to `topics: list[str]` with validation (1-10 topics), maintain backward compatibility
- [ ] **Task 1.4**: Create new response models (`src/ragster/models.py`)
  - ResearchJobResponse, JobStatusResponse, MultiTopicResponse with progress tracking
- [ ] **Task 2.1**: Build background task infrastructure (`src/ragster/background_processor.py`)
  - Async task orchestration, cancellation mechanisms, progress callbacks
- [ ] **Task 2.2**: Implement multi-topic processing strategies
  - Sequential processing (≤2 topics), parallel processing (3+ topics), intelligent strategy selection
- [ ] **Task 2.3**: Enhance TopicProcessor with progress callbacks
  - Add optional progress_callback parameter, emit progress events, maintain compatibility
- [ ] **Task 3.1**: Add global job concurrency limiter
  - MAX_CONCURRENT_RESEARCH_JOBS setting, job queue mechanism, fair scheduling
- [ ] **Task 3.2**: Implement adaptive API rate limiting
  - MultiTopicResourceManager class, dynamic semaphore sizing, adaptive batch sizes
- [ ] **Task 3.3**: Build job result caching system
  - Topic hash-based caching, TTL expiration, cache hit detection and reuse
- [ ] **Task 4.1**: Implement `get_research_status` tool
  - New MCP tool registration, GetJobStatusArgs model, real-time progress reporting
- [ ] **Task 4.2**: Implement `list_research_jobs` tool  
  - Job listing with status filtering, pagination, summary statistics
- [ ] **Task 4.3**: Implement `cancel_research_job` tool
  - Graceful task cancellation, partial result preservation, resource cleanup
- [ ] **Task 5.1**: Refactor `research_topic` to async pattern
  - Immediate return with job ID, background task creation, updated tool description
- [ ] **Task 5.2**: Integrate JobManager with AppContext and server
  - Add JobManager to AppContext, lifecycle management, job recovery for restarts
- [ ] **Task 6.1**: Add new configuration options
  - MAX_CONCURRENT_RESEARCH_JOBS, JOB_RETENTION_HOURS, JOB_CACHE_TTL_HOURS, MAX_TOPICS_PER_JOB
- [ ] **Task 6.2**: Implement job cleanup and maintenance
  - Periodic cleanup task, configurable retention policies, memory monitoring
- [ ] **Task 7.1**: Create unit tests for job management
  - JobManager CRUD tests, state transition tests, concurrency limit tests
- [ ] **Task 7.2**: Build integration tests for multi-topic processing
  - Test 1/3/10 topic scenarios, partial failure handling, performance validation
- [ ] **Task 7.3**: Implement load testing and performance validation
  - Multiple concurrent job tests, memory profiling, API rate limit compliance
- [ ] **Task 8.1**: Update documentation for async workflows
  - README.md examples, CLAUDE.md architecture updates, API documentation
- [ ] **Task 8.2**: Create example scripts and workflows
  - Async multi-topic examples, job monitoring examples, performance comparisons

## Task Details Reference

### Task 1.1: Job Management Data Models

**File**: `src/ragster/job_models.py`
**Components**:

- `JobStatus` enum: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
- `TopicProgress` dataclass: per-topic tracking with jina_status, perplexity_status, firecrawl_status, urls_found/processed, errors
- `ResearchJob` dataclass: job_id, topics, status, timestamps, progress dict, results, error
- `JobMetrics` dataclass: performance tracking and analytics

### Task 1.2: JobManager Core Class  

**File**: `src/ragster/job_manager.py`
**Features**:

- Thread-safe job storage using asyncio.Lock
- CRUD operations: create_job, get_job, update_job_status, delete_job
- State transition methods with logging
- Task tracking and cancellation support
- Memory-efficient storage with configurable retention

### Task 1.3: Multi-Topic Tool Arguments

**Changes**:

- LoadTopicToolArgs: change from `topic: str` to `topics: list[str]`
- Validation: 1-10 topics, non-empty strings, strip whitespace
- Backward compatibility: single topic workflows continue working

### Task 1.4: Async Response Models

**Models**:

- `ResearchJobResponse`: job_id, status, topics, message, created_at, estimated_completion_time
- `JobStatusResponse`: job_id, status, topics, overall_progress, topic_progress dict, results, error
- `MultiTopicResponse`: message, topics, total_urls_processed, successful_topics, failed_topics, topic_results dict

### Task 2.1: Background Task Infrastructure

**File**: `src/ragster/background_processor.py`
**Features**:

- Async task orchestration with proper exception handling
- Task cancellation and cleanup mechanisms  
- Progress callback system for real-time updates
- Task lifecycle management (start, monitor, cleanup)

### Task 2.2: Multi-Topic Processing Strategies

**Strategies**:

- Sequential: ≤2 topics (API rate limit friendly)
- Parallel: 3+ topics (with intelligent throttling)
- Strategy selection based on topic count and current system load
- Per-topic progress tracking and partial failure handling

### Task 3.1: Global Concurrency Limiter

**Features**:

- MAX_CONCURRENT_RESEARCH_JOBS setting (default: 3)
- Queue mechanism: PENDING → RUNNING when slots available
- Fair scheduling: FIFO with optional priority
- Automatic job promotion when slots free up

### Task 3.2: Adaptive Rate Limiting

**Components**:

- `MultiTopicResourceManager` class
- Dynamic Firecrawl semaphore sizing based on topic count
- Adaptive Jina search batch sizes
- Per-topic resource allocation balancing

### Task 3.3: Result Caching System

**Features**:

- Topic hash-based cache keys
- TTL-based expiration (24-48 hours configurable)
- Cache hit detection and intelligent result reuse
- Cache invalidation on error scenarios
- Memory-efficient LRU eviction

### Task 5.1: Async research_topic Implementation

**Changes**:

- Return ResearchJobResponse immediately (sub-second response)
- Create background task with job tracking
- Updated tool description for async behavior
- Proper error handling for job creation failures

### Task 5.2: AppContext Integration

**Integration Points**:

- Add JobManager to AppContext in `server.py`
- Lifecycle management in app startup/shutdown
- Job state persistence for server restart recovery
- Graceful shutdown with job state preservation

## Implementation Timeline

**Week 1**: Phase 1 (Infrastructure) + Phase 2 (Background Processing)
**Week 2**: Phase 3 (Resource Management) + Phase 4 (New Tools)
**Week 3**: Phase 5 (Core Integration) + Phase 6 (Configuration)
**Week 4**: Phase 7 (Testing) + Phase 8 (Documentation)

## Risk Mitigation

- **API Rate Limits**: Adaptive throttling and intelligent batching
- **Memory Usage**: Automatic job cleanup and result caching
- **Reliability**: Comprehensive error handling and graceful degradation
- **Compatibility**: Careful API design to maintain backward compatibility
