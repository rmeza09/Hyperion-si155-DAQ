import asyncio
from hyperion import AsyncHyperion

# Replace with your interrogator's IP
interrogator = AsyncHyperion("10.0.0.55")

async def main():
    try:
        scan_speed = await interrogator.get_laser_scan_speed()  # Await the async function
        print(f"Current laser scan speed: {scan_speed} Hz")
    except Exception as e:
        print(f"Failed to get scan speed: {e}")

# Run the async function
asyncio.run(main())
