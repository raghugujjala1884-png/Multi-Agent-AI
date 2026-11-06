import asyncio

from server import mcp


async def main():
    await mcp.run_streamable_http_async(
        host="127.0.0.1",
        port=8000
    )


if __name__ == "__main__":
    asyncio.run(main())