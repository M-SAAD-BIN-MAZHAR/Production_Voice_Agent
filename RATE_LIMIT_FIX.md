# OpenAI Rate Limit Fix

## Current Issue
Your OpenAI account shows: **3 requests/min limit**

Even though you have $500 in credits, OpenAI rate limits are based on **usage tier**, not credit balance.

## Error Message
```
Rate limit reached for tts-1 in organization org-OEo2ju7Jbq88XvRWyYfcJq2m on requests per min (RPM): 
Limit 3, Used 3, Requested 1
```

## How to Fix

### Option 1: Upgrade Usage Tier (RECOMMENDED)
1. Go to https://platform.openai.com/account/limits
2. Check your current tier (likely "Free" or "Tier 1")
3. To increase tier, you need to:
   - Make more API calls over time (automatic tier upgrade)
   - Wait for tier upgrade (happens automatically after $5-50 in usage)
   
**Tier Limits:**
- Free: 3 requests/min
- Tier 1: 60 requests/min (after $5 spent)
- Tier 2: 3,500 requests/min (after $50 spent)
- Tier 3: 10,000 requests/min (after $1,000 spent)

### Option 2: Use Code Optimizations (DONE)
I've optimized the code to:
- ✅ Use single TTS request instead of 3+ parallel requests
- ✅ Reduce VAD logging spam
- ✅ Skip rate limit delay when queue is empty
- ✅ Cache responses aggressively

### Option 3: Wait Between Queries
- Wait 20-30 seconds between questions
- Use cached responses when possible

## Current Optimizations Applied

1. **Single TTS Request**: Changed from parallel sentence-by-sentence to single request
   - Before: 3+ API calls per response
   - After: 1 API call per response
   
2. **Reduced Logging**: Removed excessive "Speech started" logs

3. **Smart Delay**: Only delays when no queries are pending

## Check Your Tier
Run this to check your current tier:
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Or visit: https://platform.openai.com/account/limits
