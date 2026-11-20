$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class GestureMouseNative {
    [DllImport("user32.dll", SetLastError = true)]
    public static extern bool SetCursorPos(int x, int y);

    [DllImport("user32.dll")]
    public static extern int GetSystemMetrics(int index);

    [DllImport("user32.dll")]
    public static extern short GetAsyncKeyState(int virtualKey);

    [DllImport("user32.dll")]
    public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, System.UIntPtr extraInfo);
}
"@

$width = [GestureMouseNative]::GetSystemMetrics(0)
$height = [GestureMouseNative]::GetSystemMetrics(1)
[Console]::Out.WriteLine("READY|$width|$height")
[Console]::Out.Flush()

while ($true) {
    $line = [Console]::In.ReadLine()
    if ($null -eq $line) {
        break
    }

    $parts = $line.Split(" ")
    switch ($parts[0]) {
        "M" {
            if ($parts.Length -ne 3) {
                throw "Invalid move command."
            }
            $x = [int]::Parse($parts[1], [Globalization.CultureInfo]::InvariantCulture)
            $y = [int]::Parse($parts[2], [Globalization.CultureInfo]::InvariantCulture)
            if (-not [GestureMouseNative]::SetCursorPos($x, $y)) {
                throw "Windows could not move the cursor."
            }
        }
        "C" {
            [GestureMouseNative]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
            [GestureMouseNative]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
        }
        "K" {
            $ctrlPressed = ([GestureMouseNative]::GetAsyncKeyState(0x11) -band 0x8000) -ne 0
            [Console]::Out.WriteLine("CTRL|{0}" -f [int]$ctrlPressed)
            [Console]::Out.Flush()
        }
        "Q" {
            break
        }
        default {
            throw "Unknown mouse command."
        }
    }
    if ($parts[0] -eq "Q") {
        break
    }
}
