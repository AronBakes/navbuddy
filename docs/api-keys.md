# API Keys

## Google Maps Platform (required)

NavBuddy uses three Google Maps APIs:

| API | Used by | Pricing |
|-----|---------|---------|
| **Street View Static API** | `navbuddy setup`, `navbuddy generate` (frame downloads) | ~$7 / 1,000 requests |
| **Directions API** | `navbuddy generate`, `navbuddy route` (routing) | ~$5 / 1,000 requests |
| **Geocoding API** | `navbuddy geocode`, address-based route generation | ~$5 / 1,000 requests |

### Getting a key

1. Go to [Google Cloud Console > Credentials](https://console.cloud.google.com/apis/credentials)
2. Create a project (or select an existing one)
3. Click **Create Credentials > API key**
4. Enable the three APIs above:
   - APIs & Services > Library > search for each API name > **Enable**
5. (Recommended) Restrict the key to only these three APIs under **API restrictions**

### Cost estimates

| Task | Estimated cost |
|------|---------------|
| NavBuddy-100 setup (100 frames) | ~$0.70 |
| NavBuddy-100 setup with `sparse4` (389 frames) | ~$2.73 |
| Custom 10-step route with `sparse4` | ~$0.28 |
| `navbuddy route` (routing only, no frames) | ~$0.005 |

Costs scale linearly with step count and frame profile. Google offers a $200/month free tier for Maps Platform.

## How NavBuddy stores keys

Running `navbuddy setup` saves keys to a `.env` file in the project root. You can also set environment variables directly:

```bash
export GOOGLE_MAPS_API_KEY="your-key-here"
export OPENROUTER_API_KEY="your-key-here"   # optional
export HF_TOKEN="your-token-here"            # optional
```

The `.env` file uses the same variable names. NavBuddy loads it automatically via `python-dotenv`.

## Optional keys

### OpenRouter API key

**Variable:** `OPENROUTER_API_KEY`

Required for running VLM inference via `navbuddy evaluate`. Supports models like GPT-4o, Claude, Gemini, and open-weight models through a unified API.

Get a key at: [https://openrouter.ai/keys](https://openrouter.ai/keys)

### HuggingFace token

**Variable:** `HF_TOKEN`

Required only for gated models (e.g., Gemma, Llama) that need access approval on HuggingFace.

Get a token at: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
