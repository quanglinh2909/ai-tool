import os
import platform
import subprocess
import sys

from app.constants.platform_enum import PlatformEnum


def detect_os():
    """
    Hàm kiểm tra và xác định hệ điều hành: Windows, Ubuntu, hay Orange Pi

    Returns:
        dict: Thông tin về hệ điều hành, bao gồm:
              - os_type: Loại hệ điều hành chung (Windows, Linux, macOS, etc)
              - os_name: Tên cụ thể của hệ điều hành
              - is_windows: True nếu là Windows
              - is_ubuntu: True nếu là Ubuntu
              - is_orangepi: True nếu là Orange Pi
              - version: Phiên bản hệ điều hành
              - architecture: Kiến trúc hệ thống (32-bit/64-bit)
              - details: Thông tin chi tiết bổ sung
    """
    result = {
        "os_type": platform.system(),
        "os_name": "Unknown",
        "is_windows": False,
        "is_ubuntu": False,
        "is_orangepi": False,
        "is_orangepi_max": False,
        "version": "",
        "architecture": platform.architecture()[0],
        "details": {}
    }

    # Kiểm tra Windows
    if platform.system() == "Windows":
        result["os_name"] = f"Windows {platform.release()}"
        result["is_windows"] = True
        result["version"] = platform.version()
        result["details"]["edition"] = platform.win32_edition() if hasattr(platform, "win32_edition") else "Unknown"

    # Kiểm tra Linux
    elif platform.system() == "Linux":
        # Tìm kiếm thông tin phân phối Linux
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                lines = f.readlines()
                os_info = {}
                for line in lines:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        os_info[key] = value.strip('"')

                if "NAME" in os_info:
                    result["os_name"] = os_info["NAME"]
                if "VERSION_ID" in os_info:
                    result["version"] = os_info["VERSION_ID"]
                if "PRETTY_NAME" in os_info:
                    result["details"]["full_name"] = os_info["PRETTY_NAME"]

        # Kiểm tra Ubuntu cụ thể
        is_ubuntu = False
        try:
            with open("/etc/lsb-release") as f:
                if "Ubuntu" in f.read():
                    is_ubuntu = True
        except:
            # Thử cách khác nếu file không tồn tại
            try:
                distro_info = subprocess.check_output("lsb_release -a", shell=True, universal_newlines=True)
                if "Ubuntu" in distro_info:
                    is_ubuntu = True
            except:
                pass

        if is_ubuntu or (result["os_name"] and "Ubuntu" in result["os_name"]):
            result["is_ubuntu"] = True

        # Kiểm tra Orange Pi (thường chạy trên ARM và có thông tin trong model name)
        is_orangepi = False
        is_orangepi_max = False
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
                if any(x in cpuinfo.lower() for x in ["orangepi", "orange pi" "h3", "h5", "h6", "allwinner"]):
                    is_orangepi = True

                if any(x in cpuinfo.lower() for x in ["opi 5 max"]):
                    is_orangepi_max = True
        except:
            pass

        # Kiểm tra thêm model hardware
        try:
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model") as f:
                    model = f.read()
                    if "Orange Pi".lower() in model.lower():
                        is_orangepi = True

                    if "opi 5 max".lower() in model.lower():
                        is_orangepi_max = True

                    result["details"]["hardware_model"] = model.strip('\0')

        except:
            pass

        # Kiểm tra board hardware qua các command phổ biến
        try:
            board_info = subprocess.check_output(
                "cat /proc/device-tree/model 2>/dev/null || cat /sys/firmware/devicetree/base/model 2>/dev/null || (which armbian-config > /dev/null && armbian-config -s)",
                shell=True, universal_newlines=True)
            if "Orange Pi".lower() in board_info :
                is_orangepi = True
            if  "opi 5 max".lower() in board_info:
                is_orangepi_max = True
            result["details"]["board_info"] = board_info.strip()
        except:
            pass

        result["is_orangepi"] = is_orangepi
        result["is_orangepi_max"] = is_orangepi_max

    # Kiểm tra macOS
    elif platform.system() == "Darwin":
        mac_ver = platform.mac_ver()
        result["os_name"] = f"macOS {mac_ver[0]}"
        result["version"] = mac_ver[0]

    # Thêm thông tin chi tiết về CPU
    try:
        if platform.system() == "Windows":
            import winreg
            reg_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            processor_name = winreg.QueryValueEx(reg_key, "ProcessorNameString")[0]
            result["details"]["cpu"] = processor_name
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                cpu_info = f.read()
                for line in cpu_info.split("\n"):
                    if "model name" in line:
                        result["details"]["cpu"] = line.split(":")[1].strip()
                        break
    except:
        result["details"]["cpu"] = platform.processor()

    # Thêm thông tin về bộ nhớ RAM
    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_mem = ctypes.c_ulong()
            kernel32.GlobalMemoryStatus(ctypes.byref(c_mem))
            result["details"]["ram"] = f"{c_mem.value / 1024 / 1024:.2f} MB"
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        memory = int(line.split()[1])
                        result["details"]["ram"] = f"{memory / 1024:.2f} MB"
                        break
    except:
        pass

    return result


# Hàm đơn giản hóa để chỉ trả về tên OS
def get_os_name():
    os_info = detect_os()
    if os_info["is_windows"]:
        return PlatformEnum.WINDOWS
    elif os_info["is_orangepi_max"]:
        return PlatformEnum.ORANGE_PI_MAX
    elif os_info["is_orangepi"]:
        return PlatformEnum.ORANGE_PI
    elif os_info["is_ubuntu"]:
        return PlatformEnum.UBUNTU

    else:
        return PlatformEnum.UNKNOWN


# Chạy kiểm tra nếu script được gọi trực tiếp
if __name__ == "__main__":
    os_info = detect_os()
    print(f"Thông tin hệ điều hành:")
    print(f"- Loại hệ điều hành: {os_info['os_type']}")
    print(f"- Tên hệ điều hành: {os_info['os_name']}")
    print(f"- Phiên bản: {os_info['version']}")
    print(f"- Kiến trúc: {os_info['architecture']}")
    print(f"- Là Windows: {'Có' if os_info['is_windows'] else 'Không'}")
    print(f"- Là Ubuntu: {'Có' if os_info['is_ubuntu'] else 'Không'}")
    print(f"- Là Orange Pi: {'Có' if os_info['is_orangepi'] else 'Không'}")

    print("\nThông tin chi tiết:")
    for key, value in os_info["details"].items():
        print(f"- {key}: {value}")

    print(f"\nTóm tắt: Hệ điều hành của bạn là {get_os_name()}")