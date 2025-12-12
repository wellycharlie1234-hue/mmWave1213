
from typing import Any, Union, Optional, Sequence, Tuple, Protocol
import numpy as np

class IKSOCIntegration(Protocol):
    def getLibVersion(self)->str:
        '''
        Get C# Lib version.
        :return: [Customer ID (00000 ~ 65535)]-[model ID (001 ~ 255)]-v[major version].[patches]
        '''
        ...

    def getDeviceInfo(self, comport_type:int=0)->str:
        '''
        Get connection information.

        0: 'Unknow' ,
        1: 'Cypress' ,
        2: 'VComPort' ,
        3: 'I2C'

        :param comport_type: Comport type number.
        :return: ComPort information
        '''
        ...

    def getSN(self)->str:

        '''
        Get series number.
        '''
        ...

    def outputDebugview(self, msg:str="", to_log_file:bool=False):
        '''
        Output message to debug view.

        :param msg: Message to ouput on debug view.
        :param isWriteLog: Save to log file.
        :return:
        '''
        ...

    def switchLogMode(self, is_print=False, debug_view=False, output_to_File=False):
        '''
        Switch log mode.

        :param Isprint: Print to console.
        :param DebugView: Output to debug view.
        :param OutputToFile: Output to log file.
        '''
        ...

    def connectDevice(self, device:int)->None:
        '''
        Connect to device.

        :param device: Device number.
        '''
        ...

    def closeDevice(self):
        '''
        Close device.
        '''
        ...

    def readHWRegister(self, addr:int, num_of_reg:int=1)->Sequence[int]:
        '''
        Read hardware register.

        :param addr: Register address.
        :param num_of_reg: Number of register.
        :return: Register value sequence.
        '''
        ...

    def writeHWRegister(self, addr:int, val_ary:Sequence[int])->None:
        '''
        Write hardware register.

        :param addr: Register address.
        :param val_ary: Register value sequence.
        '''
        ...

    def switchSPIChannel(self, mode:int)->int:
        '''
        Switch SPI channel.

        :param mode: 0: SPI0, 1: SPI1
        :return: kkt class state

        '''
        ...

    def readRFICRegister(self, addr:int)->int:
        '''
        Read RFIC register.

        :param addr: Register address, length is 2 bytes ex. 0x1234.
        :return: Register value, length is 2 bytes ex. 0x5678.
        '''
        ...

    def writeRFICRegister(self, addr:int, val:int):
        '''
        Write RFIC register.

        :param addr: Register address, length is 2 bytes ex. 0x1234.
        :param val: Register value, length is 2 bytes ex. 0x5678.
        '''
        ...

    def getAllResults(self)->Tuple[int, list, list, list, list, list, int, list]:
        '''
        Get HW results using FW cmd "0xA0".

        :return:
        '''
        ...

    def getAutoPowerStateMachine(self)->int:
        '''
        Get auto power state machine.

        :return:

        '''

    def setAutoPowerStateMachine(self, state:int):
        '''
        Set auto power state machine.

        :param state:
        :return:
        '''
        ...

    def receiveAllData_list(self)->list:
        '''
        Get HW results using FW cmd "0xA0".

        :return:
        '''
        ...

    def getGestureResult(self)->Tuple[int, np.ndarray, np.ndarray]:
        '''
        Read registers after softmax interrupt to get HW results and clear softmax interrupt.

        :return: ( gesture number, axis(x, y, z), probability )
        '''
        ...

    def setRFICScript(self, filename:str, compare:bool=False, ignore_addr_list:list=None):
        '''
        Set RFIC script.

        :param filename: RFIC script file name.
        :param compare: Compare with current RFIC script.
        :param ignore_addr_list: Ignore address list.
        '''
        ...

    def setAIWeightScript(self, filename:str, compare:bool=True):
        '''
        Set AI weight script.

        :param filename: AI weight script file name.
        :param compare: Compare with current AI weight script.

        '''

    def startMassDataBuf_RAW(self, buf_size:int, delay_ms:int, chirps:int=32):
        '''
        Start mass data buffer for RAW data.

        :param buf_size: Buffer size.
        :param delay_ms: Delay time (ms).
        :param chirps: Chirps.
        '''

    def startMassDataBuf_RDI(self, buf_size:int, delay_ms:int):
        '''
        Start mass data buffer for RDI data.

        :param buf_size: Buffer size.
        :param delay_ms: Delay time (ms).
        '''

    def stopMassDataBuf(self):
        '''
        Stop mass data buffer.
        '''

    def getMassDataBuf(self)->Tuple[int, np.ndarray, int, np.ndarray]:
        '''
        Get mass data buffer.

        :return: (ch1_frameCount, ch1 raw data, ch2_frameCount, ch2 raw data)

        '''
        ...

    def getMassDataBuf_RDI(self)->Tuple[int, int, np.ndarray]:
        '''
        Get mass data buffer for RDI data.

        :return: (ch1_frameCount, ch2_frameCount, feature map)
        '''

    def readEFuseCmd(self, addr:int)->int:
        '''
        Read EFuse command.

        :param addr: EFuse address.
        :return: EFuse value.
        '''
        ...

    def getFWVersion(self)->str:
        '''
        Get firmware version.

        :return: Firmware version.
        '''
        ...

    def resetDevice(self):
        '''
        Reset device.

        :return: kres
        '''
        ...

    def writeRawDataToSRAM(self, channel:int, rawdata:bytearray):
        '''
        Write raw data to FPGA SRam.

        :param channel: address of RX.
        :param rawdata: array of rawdata in dtype='int16'.

        '''
        ...

    def readRangeRegisters(self, first_addr: int, last_addr: int)->np.ndarray:
        '''
        Give range of registers and get array of values.

        :param first_addr: start address
        :param last_addr: end address
        :return: Array of values or None
        '''
        ...

    def switchSoftmaxInterrupt(self, actions:int, read_interrupt:int, clear_interrupt:int, size:int, ch_of_RBank:int, reg_addrs:list, frame_setting:int):
        '''
        Enable/Disable to get address's values when interrupt go high.

        :param enable: Start or stop getting data.
        :param read_interrupt: Interrupt that waiting for.
        :param clear_interrupt: Interrupt to clean.
        :param ch_of_RBank: Bit to enable the channel of reference bank .
        :param reg_addrs: Array of address.
        :return: kres
        '''
        ...

    def switchDiagnosisInterrupt_payload(self, payload: bytearray):
        '''
        Enable/Disable to get address's values when interrupt go high.

        :param enable: Start or stop getting data.
        :param read_interrupt: Interrupt that waiting for.
        :param clear_interrupt: Interrupt 20to clean.
        :param num_of_reg: Number of register.
        :param reg_addrs: Array of address.
        :return: kres
        '''
        ...

    def switchDiagnosisInterrupt(self, enable:bool, gemmini_res:int, data_size:int, reg_addrs:np.ndarray):
        '''
        Enable/Disable to get address's values when interrupt go high.

        :param enable: Start or stop getting data.
        :param read_interrupt: Interrupt that waiting for.
        :param clear_interrupt: Interrupt 20to clean.
        :param num_of_reg: Number of register.
        :param reg_addrs: Array of address.
        :return: kres
        '''
        ...

    def getSoftmaxInterruptAsserted(self)->Optional[dict]:
        '''
        Get array of register's values when interrupt go high.

        :return: None or array of register's values.
        '''
        ...

    def getDiagnosisInterruptAsserted(self)->Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        '''
        Get array of register's values when interrupt go high.

        :return: None or array of register's values.
        '''
        ...

    def updateRFICSetting(self, path:str):
        '''
        Update RFIC setting.

        :param path: RFIC setting file path.
        '''
        ...

    def runRFICInit(self):
        '''
        Run RFIC init.
        '''
        ...

    def switchAutoPowerStateMachine(self, is_stop: bool):
        '''
        Start/Stop auto power state machine.

        :param is_stop: Start(0) or stop(1).
        '''
        ...

    def getChipID(self)->str:
        '''
        Get chip ID.

        :return: Chip ID.
        '''
        ...

    def getOldFWVersion(self)->str:
        '''
        Get old firmware version.

        :return: Old firmware version.
        '''
        ...

    def setAIWeight_bin(self, filepath:str):
        '''
        Set AI weight binary file.

        :param filepath: AI weight binary file path.
        '''
        ...

    def setUserTable_bin(self, filepath:str):
        '''
        Set user table binary file.

        :param filepath: User table binary file path.
        '''
        ...

    def controlGemmini(self, mode:int):
        '''
        Control gemmini.

        :param mode: 0: stop, 1: start, 2: reset
        '''
        ...

    def readFromFlash(self, read_start_address, read_size: int, read_align: int):
        ...

    def readFromMemery(self, read_start_address, read_size: int, read_align: int):
        ...

