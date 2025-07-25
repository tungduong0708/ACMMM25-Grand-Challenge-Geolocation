# G3 Geolocation Service

This is a containerized geolocation service based on the paper "G3: An Effective and Adaptive Framework for Worldwide Geolocalization Using Large Multi-Modality Models". The service is augmented with multilayer verification for location and evidence.

## Prerequisites

- Docker with GPU support
- NVIDIA Container Toolkit (for GPU access)
- Required API keys (see Environment Variables section)

## Quick Start

### 1. Prepare Environment File

Create a `.env` file with the following variables:

```bash
GOOGLE_CLOUD_API_KEY=your_google_cloud_api_key
GOOGLE_CSE_CX=your_google_custom_search_engine_id
SCRAPINGDOG_API_KEY=your_scrapingdog_api_key
IMGBB_API_KEY=your_imgbb_api_key
GOOGLE_APPLICATION_CREDENTIALS=/code/path/to/your/credentials.json
```

### 2. Prepare Google Cloud Credentials

Ensure you have a Google Cloud service account JSON credentials file ready for copying to the container.

### 3. Build Docker Image

```bash
docker build -t g3-geolocation .
```

### 4. Create Docker Container

```bash
docker create --name g3-container -p 80:80 --gpus=all --env-file .env g3-geolocation
```

### 5. Copy Credentials to Container

```bash
docker cp /path/to/your/credentials.json g3-container:/code/
```

### 6. Start Container

```bash
docker start g3-container
```

## Usage

Once the container is running, the service will be available at `http://localhost:80`.

### API Endpoints

- **POST** `/g3/predict` - Submit images/videos for geolocation prediction
- **GET** `/g3/openapi` - Get OpenAPI specification

### Example Request

```bash
curl -X POST "http://localhost:80/g3/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@your_image.jpg"
```

## Environment Variables

| Variable                         | Description                                | Required |
| -------------------------------- | ------------------------------------------ | -------- |
| `GOOGLE_CLOUD_API_KEY`           | Google Cloud API key for Gemini and Custom Google Search API        | Yes      |
| `GOOGLE_CSE_CX`                  | Google Custom Search Engine ID             | Yes      |
| `SCRAPINGDOG_API_KEY`            | ScrapingDog API key for web scraping       | Yes      |
| `IMGBB_API_KEY`                  | ImgBB API key for image hosting            | Yes      |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google Cloud credentials JSON file | Yes      |

## API Keys Setup

### Google Cloud API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Gemini API and Vision API
3. Create an API key in the Credentials section

### Google Custom Search Engine

1. Go to [Google Custom Search](https://cse.google.com/)
2. Create a new search engine
3. Copy the Search Engine ID (CX)

### ScrapingDog API Key

1. Sign up at [ScrapingDog](https://scrapingdog.com/)
2. Get your API key from the dashboard

### ImgBB API Key

1. Sign up at [ImgBB](https://imgbb.com/)
2. Get your API key from the API section

## Container Management

### View Logs

```bash
docker logs g3-container
```

### Stop Container

```bash
docker stop g3-container
```

### Remove Container

```bash
docker rm g3-container
```

### Remove Image

```bash
docker rmi g3-geolocation
```

## Troubleshooting

### GPU Access Issues

Ensure NVIDIA Container Toolkit is properly installed:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
```

### API Key Issues

- Verify all API keys are valid and have proper permissions
- Check that the credentials file is properly copied to the container
- Ensure the `GOOGLE_APPLICATION_CREDENTIALS` path matches the copied file location

### Memory Issues

If you encounter out-of-memory errors, consider:

- Reducing image sizes before upload
- Using a machine with more RAM/VRAM
- Adjusting batch processing parameters

## Citation

```bib
@article{jia2024g3,
  title={G3: an effective and adaptive framework for worldwide geolocalization using large multi-modality models},
  author={Jia, Pengyue and Liu, Yiding and Li, Xiaopeng and Zhao, Xiangyu and Wang, Yuhao and Du, Yantong and Han, Xiao and Wei, Xuetao and Wang, Shuaiqiang and Yin, Dawei},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={53198--53221},
  year={2024}
}
```
