import os
import inspect
import pathlib
import typing

import clr
import sys

clr.AddReference("System")

import System
from System import Array, UInt32, Int32, UInt16, Int16, String, Byte
import numpy as np

Core_PATH = pathlib.Path(__file__).parent
sys.path.append(str(Core_PATH))

from KKT_Library.LibLog import lib_logger as log

DLL_NAME = "KSOC_Lib"
DLL_PATH = None


def _find_ksoc_dll():
    """自動搜尋 KSOC_Lib.dll"""
    global DLL_PATH

    if DLL_PATH is not None and os.path.exists(DLL_PATH):
        return DLL_PATH

    # 方法 1: 從當前檔案位置往上搜尋
    current_file = pathlib.Path(__file__).resolve()
    search_paths = [
        current_file.parent,  # Integration 目錄
        current_file.parents[1],  # KKT_Library 目錄
        current_file.parents[2],  # Library 目錄
        current_file.parents[3],  # KKT_Module_Example_20240820 目錄
    ]

    # 方法 2: 加入可能的專案根目錄
    for parent in current_file.parents:
        if parent.name in ["KKT_Module_Example_20240820", "Library", "mmWave"]:
            search_paths.append(parent)

    # 去重
    search_paths = list(dict.fromkeys(search_paths))

    for base_dir in search_paths:
        log.info(f"[CSharpWrapper] 搜尋 KSOC_Lib.dll,起點:{base_dir}")

        for root, dirs, files in os.walk(base_dir):
            if "KSOC_Lib.dll" in files:
                DLL_PATH = os.path.join(root, "KSOC_Lib.dll")
                log.info(f"[CSharpWrapper] 找到 KSOC_Lib.dll:{DLL_PATH}")
                return DLL_PATH

    log.error(f"[CSharpWrapper] 無法找到 KSOC_Lib.dll，已搜尋路徑: {search_paths}")
    return None


def addReferenceLib(dll_name):
    """載入 C# DLL"""
    dll_path = _find_ksoc_dll()

    if dll_path is None or not os.path.exists(dll_path):
        error_msg = f"[CSharpWrapper] 找不到 {dll_name}.dll，請檢查:\n"
        error_msg += f"  1. DLL 檔案是否存在\n"
        error_msg += f"  2. 當前工作目錄: {os.getcwd()}\n"
        error_msg += f"  3. CSharpWrapper.py 位置: {__file__}\n"
        log.error(error_msg)
        raise FileNotFoundError(error_msg)

    dll_dir = os.path.dirname(dll_path)
    if dll_dir not in sys.path:
        sys.path.append(dll_dir)
        log.info(f"[CSharpWrapper] 已將目錄加入 sys.path: {dll_dir}")

    try:
        ksoc_lib = clr.AddReference(dll_name)
        log.info(f"[CSharpWrapper] clr.AddReference 成功 = {ksoc_lib}")
        return ksoc_lib
    except Exception as e:
        log.error(f"[CSharpWrapper] clr.AddReference 失敗: {e}")
        try:
            log.info(f"[CSharpWrapper] 嘗試使用 LoadFrom: {dll_path}")
            System.Reflection.Assembly.LoadFrom(dll_path)
            ksoc_lib = clr.AddReference(dll_name)
            log.info(f"[CSharpWrapper] LoadFrom 成功")
            return ksoc_lib
        except Exception as e2:
            log.error(f"[CSharpWrapper] LoadFrom 失敗: {e2}")
            raise


# 載入 DLL
KSOC_Lib = addReferenceLib(DLL_NAME)

# 直接從組件取得類型
log.info("[CSharpWrapper] 直接從組件取得類型...")

ksoc_assembly = None
for asm in System.AppDomain.CurrentDomain.GetAssemblies():
    if asm.GetName().Name == "KSOC_Lib":
        ksoc_assembly = asm
        break

if ksoc_assembly is None:
    raise Exception("找不到 KSOC_Lib 組件")

# 取得類型
KSOC_Integration = ksoc_assembly.GetType("KSOC_Lib.KSOC_Integration")
Kit_Communication_Type = ksoc_assembly.GetType("KSOC_Lib.Kit_Communication_Type")
KKT_USB_Device_Index = ksoc_assembly.GetType("KSOC_Lib.KKT_USB_Device_Index")
MassDataProc_Type = ksoc_assembly.GetType("KSOC_Lib.MassDataProc_Type")

log.info("[CSharpWrapper] 成功取得所有類型")


def printEnvironment():
    '''print .net runtime version and current environment'''
    print(System.Environment.Version)
    print('---------------------')
    for p in sys.path:
        print(p)
    print('---------------------')


def printCLRInfo():
    '''列印程式集'''
    lt = clr.ListAssemblies(False)
    for i in range(lt.Length):
        print('%d = %s' % (i, lt[i]))


class KKTLibException(Exception):
    def __init__(self, function_name, kres, *args, **kwargs):
        msg = "[ {} ] Failure, kres = {}".format(function_name, kres)
        super(KKTLibException, self).__init__(msg, *args, **kwargs)


class KSOCIntegration:
    def __init__(self):
        # 使用 Activator 創建 C# 類別實例
        self._instance = System.Activator.CreateInstance(KSOC_Integration)

    def __getattr__(self, name):
        # 代理所有未定義的屬性到 C# 實例
        return getattr(self._instance, name)

    def getLibVersion(self) -> str:
        '''Get C# Lib version.'''
        return self._instance.GetLibVersionInfor()

    def getDeviceInfo(self, comport_type=2) -> str:
        '''Get ComPort informations.'''
        return self._instance.GetDeviceInfor(comport_type)

    def getSN(self):
        '''Get series number.'''
        kres, SN = self._instance.GetSN(None)
        if kres != 0:
            raise KKTLibException(function_name=inspect.stack()[0][3], kres=kres)
        return SN

    def outputDebugview(self, msg="", isWriteLog=False):
        '''Output message to debug view.'''
        self._instance.OutputMsgToDebugView(msg, isWriteLog)

    def switchLogMode(self, Isprint=False, DebugView=False, OutputToFile=False):
        self._instance.SwitchLogMode(Isprint, DebugView, OutputToFile)

    def connectDevice(self, device):
        # 取得枚舉值
        device_index = KKT_USB_Device_Index.GetField("KKT_USB_Device_0").GetValue(None)
        kres = self._instance.ConnectUsbDevice(device, device_index)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def closeDevice(self):
        kres = self._instance.CloseUsbDevice()

    def readHWRegister(self, addr, num_of_reg=1):
        kres, val = self._instance.Device_Read_Reg(addr, num_of_reg, None)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return np.asarray(list(val), dtype='uint32')

    def writeHWRegister(self, addr, val_ary):
        new_val_ary = Array[UInt32](val_ary)
        kres = self._instance.Device_Write_Reg(addr, new_val_ary)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def switchSPIChannel(self, mode):
        assert mode in [0, 1]
        kres = self._instance.SwitchSPIChannel(mode)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return kres

    def readRFICRegister(self, addr):
        kres, val = self._instance.RFIC_Cmd_Read(UInt16(addr), UInt16(0))
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return val

    def writeRFICRegister(self, addr, val):
        kres = self._instance.RFIC_Cmd_Write(UInt16(addr), UInt16(val))
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def getAllResults(self):
        kres, ges, axes_ary, softmax_ary, sia_ges, sia_softmax_ary, motion_rssi, motion_rssi_ary = self._instance.RegGet_AllResult(
            UInt32(0), None, None, UInt32(0), None, UInt32(0), None)
        if kres == 0:
            return ges, axes_ary, softmax_ary, sia_ges, sia_softmax_ary, motion_rssi, motion_rssi_ary

    def getAutoPowerStateMachine(self):
        kres, PowerSateOrAck = self._instance.GetAutoPowerStateMachine(Int32(0))
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return PowerSateOrAck

    def setAutoPowerStateMachine(self, PowerState: int):
        kres = self._instance.SetAutoPowerStateMachine(PowerState)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return kres

    def receiveAllData_list(self):
        '''Get HW results using FW cmd "0xA0".'''
        kres, data = self._instance.ReceiveAllDataAsList(None)
        if kres != 0:
            return None
        data = list(data)
        if data[0] == -1:
            return None
        if data[2] is not None:
            data[2] = np.asarray(list(data[2])).astype('int16')
        if data[3] is not None:
            data[3] = np.asarray(list(data[3])).astype('uint16')
        return data

    def getGestureResult(self):
        '''Read registers after softmax interrupt to get HW results and clear softmax interrupt.'''
        kres, ges, axes_ary, softmax_ary = self._instance.RegGet_GESTURE(UInt32(0), None, None)
        if kres == 0:
            return ges, np.asarray(list(axes_ary), dtype='int16'), np.asarray(list(softmax_ary), dtype='uint16')

    def setRFICScript(self, filename, compare=False, ignoreAddrList=None):
        kres = self._instance.SetScript_Rfic(filename, compare, ignoreAddrList)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def setAIWeightScript(self, filename, compare=True):
        kres = self._instance.SetScript_AIWeight(filename, compare)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def startMassDataBuf_RAW(self, buf_size, delay_ms, chirps=32):
        # 取得枚舉值
        if chirps == 16:
            mass_proc_type = MassDataProc_Type.GetField("MDP_Normal_16").GetValue(None)
        elif chirps == 32:
            mass_proc_type = MassDataProc_Type.GetField("MDP_Normal_32").GetValue(None)
        elif chirps == 64:
            mass_proc_type = MassDataProc_Type.GetField("MDP_Normal_64").GetValue(None)
        kres = self._instance.MassDataBuffer_Start_Setting(mass_proc_type, UInt32(buf_size), UInt32(delay_ms))
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def startMassDataBuf_RDI(self, buf_size, delay_ms):
        mass_proc_type = MassDataProc_Type.GetField("MDP_Normal_RDI").GetValue(None)
        kres = self._instance.MassDataBuffer_Start_Setting(mass_proc_type, UInt32(buf_size), UInt32(delay_ms))
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def stopMassDataBuf(self):
        kres = self._instance.MassDataBuffer_Stop()
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def getMassDataBuf(self):
        kres, ch1_frameCount, ch1, ch2_frameCount, ch2 = self._instance.MassDataBuffer_Get(UInt16(0), None, UInt16(0),
                                                                                           None)
        if kres == 0:
            return ch2_frameCount, np.asarray(list(ch1), dtype='int16'), \
                ch1_frameCount, np.asarray(list(ch2), dtype='int16')

    def getMassDataBuf_RDI(self):
        kres, ch1_frameCount, ch2_frameCount, rdi_raw = self._instance.MassDataBuffer_GetRDIRaw(UInt16(0), UInt16(0),
                                                                                                None)
        if kres == 0:
            return (ch1_frameCount, ch2_frameCount, np.asarray(list(rdi_raw), dtype='uint16'))

    def readEFuseCmd(self, addr):
        kres, outreg_val = self._instance.EFuse_Cmd_Read(addr, None)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return outreg_val[0]

    def getFWVersion(self):
        kres, version = self._instance.GetFWVersion(None)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return version

    def resetDevice(self):
        '''Reset device.'''
        kres = self._instance.Reset_Device()
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def writeRawDataToSRAM(self, channel, rawdata: bytearray):
        '''Write raw data to SRAM.'''
        kres = self._instance.Write_Raw_to_SRAM(channel, rawdata)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def readRangeRegisters(self, first_addr: int, last_addr: int):
        '''Give range of registers and get array of values.'''
        kres, val_arry = self._instance.Read_Register_Range(first_addr, last_addr, None)
        if kres == 0:
            return np.asarray(list(val_arry)).astype('int32')

    def switchSoftmaxInterrupt(self, enable, read_interrupt, clear_interrupt, size, ch_of_RBank, reg_addrs,
                               frame_setting):
        '''Enable/Disable to get address's values when interrupt go high.'''
        kres = self._instance.SwitchSoftMaxInterruptAsserted(enable, read_interrupt, clear_interrupt, size,
                                                             ch_of_RBank, reg_addrs, frame_setting)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def switchDiagnosisInterrupt_payload(self, payload: bytearray):
        '''Enable/Disable to get address's values when interrupt go high.'''
        kres = self._instance.SwitchDiagnosisInterruptAsserted(payload)
        if kres != 0:
            raise Exception("[{}] Failure, kres = {}".format(inspect.stack()[0][3], kres))

    def switchDiagnosisInterrupt(self, enable, gemmini_res, data_size, reg_addrs):
        '''Enable/Disable to get address's values when interrupt go high.'''
        kres = self._instance.SwitchDiagnosisInterruptAsserted(enable, gemmini_res, data_size, reg_addrs)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def getSoftmaxInterruptAsserted(self, take_first_frame) -> typing.Optional[dict]:
        '''Get array of register's values when interrupt go high.'''
        kres, reg_vals = self._instance.GetSoftMaxInterruptAssertedRegValues(None, take_first_frame)
        if kres != 0:
            return None
        return {v.Key: bytearray(v.Value) for v in reg_vals}

    def getDiagnosisInterruptAsserted(self, take_first_frame):
        '''Get array of register's values when interrupt go high.'''
        kres, raw, diagnosis, AGC, Motion, InterferenceBank = self._instance.GetDiagnosisInterruptAssertedRegValues(
            None, None, None, None, None, take_first_frame)
        if kres != 0:
            return None
        if raw is not None:
            raw = np.asarray(list(raw), dtype='int16')
        if diagnosis is not None:
            diagnosis = list(diagnosis)
        if AGC is not None:
            AGC = bytearray(AGC)
            AGC = np.asarray(AGC, dtype='int8')
        if Motion is not None:
            Motion = np.asarray(list(Motion), dtype='int16')
        if InterferenceBank is not None:
            InterferenceBank = bytearray(InterferenceBank)
            InterferenceBank = np.frombuffer(InterferenceBank, dtype=np.uint32)
        return raw, diagnosis, AGC, Motion, InterferenceBank

    def updateRFICSetting(self, path):
        kres = self._instance.UpdateRFICSetting(path)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def runRFICInit(self):
        kres = self._instance.RunRFICInit()
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def switchAutoPowerStateMachine(self, IsStop: bool):
        kres = self._instance.SwitchAutoPowerStateMachine(IsStop)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def getChipID(self):
        kres, chip_id = self._instance.GetChipID(None)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return chip_id

    def getOldFWVersion(self):
        kres, old_ver = self._instance.GetOldVersion(None)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return old_ver

    def setAIWeight_bin(self, filepath):
        kres = self._instance.Set_AIWeight_Bin(filepath)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def setUserTable_bin(self, filepath):
        kres = self._instance.Set_UserTable_Bin(filepath)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def controlGemmini(self, mode):
        kres = self._instance.Control_Gemmini(mode)
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)

    def initRFIC(self):
        kres = self._instance.InitRFIC()
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return kres

    def initSIC(self):
        kres = self._instance.SICInit()
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return kres

    def setDigiParam0(self):
        kres = self._instance.SetDigiParam0()
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return kres

    def getRXPhaseOffset(self):
        kres, rx_13_offset, rx_23_offset, temperature = self._instance.GetRXPhaseOffset(Byte(0), Byte(0), Byte(0))
        if kres != 0:
            raise KKTLibException(inspect.stack()[0][3], kres)
        return int(rx_13_offset), int(rx_23_offset), int(temperature)

    def readFromFlash(self, read_start_address: UInt32, read_size: int, read_align: int):
        kres, data = self._instance.ReadDataFromFlash(read_start_address, read_size, read_align, None)
        if kres != 0:
            return None
        return data

    def readFromMemery(self, read_start_address: UInt32, read_size: int, read_align: int):
        kres, data = self._instance.ReadDataFromMemery(read_start_address, read_size, read_align, None)
        if kres != 0:
            return None
        return data


if __name__ == '__main__':
    import time

    # 選擇模式
    print("\n" + "=" * 60)
    print("KKT mmWave 測試程式")
    print("=" * 60)
    print("\n請選擇模式:")
    print("  1 - 基本測試（顯示版本和設備資訊）")
    print("  2 - 動作辨識（即時偵測手勢）")
    print("  3 - 原始數據收集")
    print("  0 - 退出")

    try:
        choice = input("\n請輸入選項 (預設為 1): ").strip()
        if not choice:
            choice = "1"
    except:
        choice = "1"

    if choice == "0":
        print("退出程式")
        sys.exit(0)

    # 模式 1: 基本測試
    if choice == "1":
        print("\n" + "=" * 60)
        print("基本測試模式")
        print("=" * 60)
        printEnvironment()
        printCLRInfo()
        k = KSOCIntegration()
        print(f"\nLibrary Version: {k.getLibVersion()}")
        print(f"Device Info: {k.getDeviceInfo()}")
        print("\nDone")

    # 模式 2: 動作辨識
    elif choice == "2":
        print("\n" + "=" * 60)
        print("動作辨識模式")
        print("=" * 60)

        k = None
        try:
            # 初始化
            print("\n[1] 初始化系統...")
            k = KSOCIntegration()
            print(f"    Library 版本: {k.getLibVersion()}")
            print(f"    設備資訊: {k.getDeviceInfo()}")

            # 連接設備
            print("\n[2] 連接設備...")
            k.connectDevice(2)  # 2 = VComPort
            print("    ✓ 設備連接成功")

            # 取得設備資訊
            try:
                sn = k.getSN()
                print(f"    序號: {sn}")
            except:
                pass

            try:
                fw_ver = k.getFWVersion()
                print(f"    韌體版本: {fw_ver}")
            except:
                pass

            # 初始化 RFIC
            print("\n[3] 初始化射頻晶片...")
            k.initRFIC()
            print("    ✓ RFIC 初始化成功")

            # 等待系統穩定
            print("\n[4] 等待系統穩定...")
            time.sleep(2)
            print("    ✓ 系統就緒")

            # 開始動作辨識
            print("\n" + "=" * 60)
            print("開始動作辨識 (按 Ctrl+C 停止)")
            print("=" * 60)
            print("\n動作代碼說明:")
            print("  0 = 無動作")
            print("  1-6 = 各種手勢動作")
            print()

            gesture_count = 0
            loop_count = 0
            last_gesture = 0

            while True:
                loop_count += 1

                try:
                    # 取得辨識結果
                    results = k.getAllResults()

                    if results:
                        ges, axes_ary, softmax_ary, sia_ges, sia_softmax_ary, motion_rssi, motion_rssi_ary = results

                        # 偵測到新動作
                        if ges != 0 and ges != last_gesture:
                            gesture_count += 1
                            last_gesture = ges

                            print(f"\n{'=' * 60}")
                            print(f"[動作 #{gesture_count}] 時間: {time.strftime('%H:%M:%S')}")
                            print(f"{'=' * 60}")
                            print(f"  動作編號: {ges}")

                            if axes_ary is not None:
                                print(f"  軸向: {axes_ary}")

                            if softmax_ary is not None:
                                max_conf = max(softmax_ary) if len(softmax_ary) > 0 else 0
                                print(f"  信心度: {max_conf}")

                            print(f"  運動強度: {motion_rssi}")
                            print(f"{'=' * 60}\n")

                        # 回到無動作狀態
                        if ges == 0:
                            last_gesture = 0

                        # 進度指示
                        if loop_count % 50 == 0:
                            print(".", end="", flush=True)

                    # 延遲
                    time.sleep(0.05)

                except KeyboardInterrupt:
                    print("\n\n使用者中斷")
                    break
                except Exception as e:
                    print(f"\n辨識錯誤: {e}")
                    time.sleep(0.5)

            print(f"\n總共偵測到 {gesture_count} 個動作")

        except Exception as e:
            print(f"\n錯誤: {e}")
            import traceback

            traceback.print_exc()

        finally:
            if k is not None:
                try:
                    print("\n[5] 關閉設備...")
                    k.closeDevice()
                    print("    ✓ 設備已關閉")
                except:
                    pass

    # 模式 3: 原始數據收集
    elif choice == "3":
        print("\n" + "=" * 60)
        print("原始數據收集模式")
        print("=" * 60)

        k = None
        try:
            k = KSOCIntegration()
            print(f"\nLibrary 版本: {k.getLibVersion()}")

            # 連接設備
            print("\n連接設備...")
            k.connectDevice(2)
            print("✓ 設備連接成功")

            # 初始化 RFIC
            print("\n初始化 RFIC...")
            k.initRFIC()
            print("✓ RFIC 初始化成功")

            # 開始收集原始數據
            print("\n開始收集原始數據...")
            buf_size = 100
            delay_ms = 10
            chirps = 32

            k.startMassDataBuf_RAW(buf_size, delay_ms, chirps)
            print(f"✓ 已啟動 (緩衝: {buf_size}, 延遲: {delay_ms}ms, Chirps: {chirps})")

            # 收集數據
            print("\n收集 10 次數據...")
            for i in range(10):
                time.sleep(0.5)
                data = k.getMassDataBuf()
                if data:
                    ch2_count, ch1_data, ch1_count, ch2_data = data
                    print(f"  [{i + 1}] CH1: {ch1_count} frames, CH2: {ch2_count} frames")

            # 停止收集
            print("\n停止數據收集...")
            k.stopMassDataBuf()
            print("✓ 數據收集已停止")

            k.closeDevice()
            print("\n✓ 測試完成")

        except Exception as e:
            print(f"\n錯誤: {e}")
            import traceback

            traceback.print_exc()

            if k is not None:
                try:
                    k.stopMassDataBuf()
                    k.closeDevice()
                except:
                    pass

    else:
        print("\n無效的選項")