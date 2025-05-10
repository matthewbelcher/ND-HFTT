import websockets
import inspect
import asyncio

print(f"Websockets version: {websockets.__version__}")
print(f"Available methods in websockets.connect: {inspect.signature(websockets.connect)}")

async def test_connect():
    try:
        # Try a simple connection test
        async with websockets.connect("wss://echo.websocket.org") as websocket:
            await websocket.send("Hello!")
            response = await websocket.recv()
            print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    asyncio.run(test_connect()) 