from starlette.applications import Starlette
from starlette.responses import StreamingResponse, Response, JSONResponse
from starlette.requests import Request
from starlette.routing import Route
import httpx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import yaml
import random
import asyncio

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
basic_config = {
    "endpoints": [
        {'api_version': '2024-05-01-preview', 'deployment': 'gpt-4-0125-Preview-spot', 'url': 'https://aims-genalign-spot-eastus.openai.azure.com/'}, 
        {'api_version': '2024-05-01-preview', 'deployment': 'gpt-4-0125-Preview-spot', 'url': 'https://aims-genalign-spot-japaneast.openai.azure.com/'}, 
        {'api_version': '2024-05-01-preview', 'deployment': 'gpt4-t-1106', 'url': 'https://aims-datagen-spot-safrican.openai.azure.com/'}, 
        {'api_version': '2024-05-01-preview', 'deployment': 'gpt-4-0125-Preview-spot', 'url': 'https://aims-genalign-spot-swedencentral.openai.azure.com/'}, 
        {'api_version': '2024-05-01-preview', 'deployment': 'gpt-4-0613-spot', 'url': 'https://aims-genalign-spot-polandcentral.openai.azure.com/'}
    ]
}
EXCLUDED_HEADERS = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']


def process_headers(request_headers, token):
    headers = {key.lower(): value for key, value in request_headers.items()}
    headers['authorization'] = f'Bearer {token}'
    headers.pop('host', None)
    headers.pop('content-length', None)  # Let httpx handle Content-Length
    headers.pop('connection', None)  # Let httpx handle Connection
    return headers


def filter_response_headers(response):
    return [(name, value) for (name, value) in response.headers.items() if name.lower() not in EXCLUDED_HEADERS]


def url_join(base, *args):
    base = base.rstrip('/')
    parts = [arg.lstrip('/') for arg in args]
    return '/'.join([base] + parts)


async def test_endpoint(client, endpoint, token):
    url = url_join(endpoint['url'], 'openai/deployments', endpoint['deployment'],  'chat/completions')
    headers = {
        "accept-encoding": "gzip, deflate",
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
    }
    content = b'{"messages": [{"role": "user", "content": "Please write a haiku"}], "model": "gpt-4"}'
    params = {"api-version": endpoint["api_version"]}

    try:
        response = await client.post(url, headers=headers, content=content, params=params)
        if response.status_code == 200:
            return True
        print(f"Status for {url}: {response.status_code}, {response.json()}")
    except Exception as e:
        print(f"Error for {url}: {repr(e)}")
    return False


async def filter_endpoints(config):
    token = token_provider()
    async with httpx.AsyncClient() as client:
        tasks = [
            test_endpoint(client, endpoint, token) for endpoint in config["endpoints"]
        ]
        results = await asyncio.gather(*tasks)

    return [endpoint for success, endpoint in zip(results, config["endpoints"]) if success]


async def lifespan(app):
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(e)
        print("Falling back to a simple config with one endpoint")
        config = basic_config

    app.state.endpoints = await filter_endpoints(config)
    if len(app.state.endpoints) == 0:
        raise RuntimeError("No valid endpoints found")
    else:
        print(f"Valid endpoints: {app.state.endpoints}")
        yield


# Proxy request handler
async def proxy(request: Request):
    path = request.path_params.get('path', '')
    data = await request.body()
    endpoint = random.choice(request.app.state.endpoints)
    headers = process_headers(request.headers, token_provider())
    azure_url = url_join(endpoint['url'], 'openai/deployments', endpoint['deployment'],  path)
    params = dict(request.query_params)
    params['api-version'] = endpoint['api_version']
    stream = 'stream' in request.query_params and request.query_params.get('stream') == 'true'

    try:
        async with httpx.AsyncClient() as client:
            if stream:
                # For streaming requests
                async with client.stream(
                    method=request.method,
                    url=azure_url,
                    headers=headers,
                    content=data,
                    params=params
                ) as response:
                    response_headers = filter_response_headers(response)

                    async def stream_content():
                        async for chunk in response.aiter_bytes():
                            yield chunk

                    return StreamingResponse(stream_content(), status_code=response.status_code, headers=dict(response_headers))
            else:
                # For non-streaming requests
                response = await client.request(
                    method=request.method,
                    url=azure_url,
                    headers=headers,
                    content=data,
                    params=params
                )
                response_headers = filter_response_headers(response)


                return Response(response.content, status_code=response.status_code, headers=dict(response_headers))

    except httpx.ReadTimeout:
        return JSONResponse(
            {"error": "Request to Azure OpenAI endpoint timed out. Please try again later."},
            status_code=504  # 504 Gateway Timeout
        )
    except Exception as e:
        return JSONResponse(
            {
                "error": "An unexpected error occurred.",
                "details": repr(e),
            },
            status_code=500  # 500 Internal Server Error
        )


routes = [
    Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]),
    Route("/", proxy, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]),
]

app = Starlette(debug=True, routes=routes, lifespan=lifespan)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=5555)
