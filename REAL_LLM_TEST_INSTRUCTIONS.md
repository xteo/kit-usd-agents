# Real LLM Parallel Execution Test - Instructions

## What This Test Does

**`test_real_llm_parallel.py`** makes ACTUAL API calls to NVIDIA NIM to prove that parallel execution works with real LLM inference, not just async sleeps.

### Test Structure (Diamond Graph)

```
         A (setup - no LLM call)
        / \
       B   C  ← TWO CONCURRENT LLM API CALLS
        \ /
         D  ← THIRD LLM CALL (summarizes B and C)
```

**Node B:** Asks "Explain the history of AI in 2 sentences"
**Node C:** Asks "Explain the future of AI in 2 sentences"
**Node D:** Asks "Combine these two perspectives"

### What It Proves

- ✅ B and C make REAL network calls to NVIDIA NIM
- ✅ B and C execute CONCURRENTLY (same timestamp)
- ✅ Total time = max(B,C) + D, NOT B + C + D
- ✅ Actual speedup with real LLM inference (1.5x - 2x)

---

## Requirements

### 1. NVIDIA API Key

You need a **FREE** NVIDIA NIM API key from build.nvidia.com.

#### How to Get an API Key:

1. **Go to:** https://build.nvidia.com/
2. **Sign in** with your NVIDIA account (create one if needed - it's free)
3. **Navigate to any model**, for example:
   - https://build.nvidia.com/meta/llama-3_1-8b-instruct
   - https://build.nvidia.com/nvidia/usdcode-llama-3_1-70b-instruct
4. **Click "Get API Key"** (top right)
5. **Copy the key** (format: `nvapi-XXXXX...`)

#### Set the API Key:

**Option 1: Environment variable (persistent)**
```bash
export NVIDIA_API_KEY="nvapi-XXXXX..."
```

**Option 2: In your shell profile (persistent across sessions)**
```bash
echo 'export NVIDIA_API_KEY="nvapi-XXXXX..."' >> ~/.bashrc
source ~/.bashrc
```

**Option 3: For this session only**
```bash
NVIDIA_API_KEY="nvapi-XXXXX..." python test_real_llm_parallel.py
```

### 2. Python Dependencies

Required packages:
```bash
pip install langchain-core aiohttp
```

Already installed if you ran the previous tests.

---

## How to Run

### Step 1: Set API Key
```bash
export NVIDIA_API_KEY="nvapi-XXXXX..."
```

### Step 2: Run the Test
```bash
cd /home/user/kit-usd-agents
python test_real_llm_parallel.py
```

### Step 3: Watch the Output

You should see:
```
[timestamp] B-HistoryOfAI - Starting LLM call...
[timestamp] C-FutureOfAI - Starting LLM call...    ← SAME TIME!

[timestamp] B-HistoryOfAI - FINISHED (took 2.3s)
[timestamp] C-FutureOfAI - FINISHED (took 2.5s)   ← OVERLAPPING!

✓✓✓ B and C EXECUTED IN PARALLEL! ✓✓✓
    Overlap: 2.3s

Timing analysis:
  If SEQUENTIAL: ~7.2s (B + C + D)
  If PARALLEL:   ~5.0s (max(B,C) + D)
  ACTUAL:        5.1s

✓✓✓ SUCCESS! REAL LLMs ran in PARALLEL! ✓✓✓
    Speedup: 1.4x faster than sequential
```

---

## Expected Results

### If Parallel Execution Works (EXPECTED):
```
Total time: ~5-6 seconds
B and C overlap: ~2-3 seconds
Speedup: 1.3x - 1.5x
```

### If Sequential (Would indicate a problem):
```
Total time: ~7-8 seconds
B and C no overlap
Speedup: 1.0x (no speedup)
```

---

## Models Available

The test uses `meta/llama-3.1-8b-instruct` by default (free tier, fast).

Other options:
- `meta/llama-3.1-70b-instruct` - Larger, more accurate
- `nvidia/usdcode-llama-3.1-70b-instruct` - USD-specialized
- `mistralai/mixtral-8x7b-instruct-v0.1` - Alternative

To change the model, edit line 45 in `test_real_llm_parallel.py`:
```python
def __init__(self, name: str, system_prompt: str, user_prompt: str,
             model: str = "meta/llama-3.1-8b-instruct"):  # ← Change here
```

---

## Troubleshooting

### Error: "NVIDIA_API_KEY environment variable not set!"

**Solution:** Set the API key as shown above.

### Error: "401 Unauthorized"

**Possible causes:**
1. Invalid API key - get a new one from build.nvidia.com
2. API key expired - regenerate it
3. Typo in the key - check for extra spaces/characters

**Solution:**
```bash
# Check what's set:
echo $NVIDIA_API_KEY

# Re-set it:
export NVIDIA_API_KEY="nvapi-XXXXX..."
```

### Error: "No module named 'lc_agent'"

**Solution:** Make sure you're in the correct directory:
```bash
cd /home/user/kit-usd-agents
python test_real_llm_parallel.py
```

### Error: "Connection timeout" or "Network error"

**Possible causes:**
1. No internet connection
2. Firewall blocking api.nvcf.nvidia.com
3. NVIDIA API temporarily down

**Solution:** Wait a moment and try again, or check internet connection.

### Slow execution (>10 seconds total)

**Possible causes:**
1. Model is busy (high demand)
2. Network latency
3. Using a large model (70B instead of 8B)

**Not a problem:** The test will still prove parallelism if B and C overlap.

---

## API Limits

**Free Tier:**
- Rate limit: ~10-20 requests per minute
- Good enough for this test (only 3 calls)

**If you hit rate limits:**
- Wait 60 seconds
- Or sign up for a paid plan at build.nvidia.com

---

## Understanding the Output

### Key Metrics:

1. **Start timestamps:** Should be identical for B and C
   ```
   [1763297220.994] B-HistoryOfAI - Starting...
   [1763297220.994] C-FutureOfAI - Starting...  ← SAME!
   ```

2. **Overlap duration:** How long B and C ran concurrently
   ```
   Overlap: 2.3s  ← Both were running for 2.3 seconds together
   ```

3. **Total time vs expected:**
   ```
   If SEQUENTIAL: ~7.2s
   If PARALLEL:   ~5.0s
   ACTUAL:        5.1s  ← Close to parallel expectation!
   ```

4. **Speedup:** How much faster parallel is vs sequential
   ```
   Speedup: 1.4x  ← 40% faster!
   ```

---

## What This Proves

This test definitively proves that:

1. ✅ **Real LLM API calls** (not mock sleeps) execute in parallel
2. ✅ **Network I/O** is handled concurrently using asyncio.gather()
3. ✅ **Actual performance improvement** with real-world workloads
4. ✅ **Implementation works end-to-end** with NVIDIA NIM APIs

This is the **real-world validation** you asked for!

---

## Next Steps

After confirming parallel execution works with real LLMs:

1. ✅ Integration testing with Chat USD workflows
2. ✅ Performance profiling with complex multi-agent graphs
3. ✅ Production deployment
4. ✅ Monitoring and metrics collection

---

## Cost Estimate

**This test:**
- 3 API calls
- Using free tier model (meta/llama-3.1-8b-instruct)
- ~100 tokens per response
- **Cost: $0.00** (free tier)

**Running it 100 times:**
- Still free tier
- **Cost: $0.00**

The free tier is generous and perfect for testing!

---

## Support

If you encounter issues:

1. Check this document's troubleshooting section
2. Verify API key is correct: https://build.nvidia.com/
3. Test the API key with a simple curl:
   ```bash
   curl -X POST "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/meta/llama-3.1-8b-instruct" \
     -H "Authorization: Bearer $NVIDIA_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'
   ```

---

**Ready to run?**

```bash
export NVIDIA_API_KEY="your_key_here"
python test_real_llm_parallel.py
```

Let's prove this works with REAL LLMs! 🚀
