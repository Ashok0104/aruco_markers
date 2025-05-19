import asyncio
from bleak import BleakClient

# Replace with your bot's MAC address
ADDRESS = "3C:84:27:C2:A0:AD"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

async def send_command(command):
    async with BleakClient(ADDRESS) as client:
        if not client.is_connected:
            print("Failed to connect")
            return
        
        print(f"Connected to {ADDRESS}")
        
        # Process each character in the command string individually
        for char in command.strip().upper():  # Ensure uppercase and no extra spaces
            if char in ['F', 'B', 'L', 'R']:
                print(f"Sending: {char}")
                await client.write_gatt_char(UART_RX_CHAR_UUID, char.encode())  # Send each as bytes
                #await asyncio.sleep(0.05)  # Small delay to allow processing (tune if necessary)
            else:
                print(f"Invalid command '{char}'. Use F, B, L, or R.")

if __name__ == "_main_":
    while True:
        cmd = input("Enter command sequence (e.g., FFRFBBL) or 'exit' to quit: ")
        if cmd.lower() == "exit":
            break
        asyncio.run(send_command(cmd))