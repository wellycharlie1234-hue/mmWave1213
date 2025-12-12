# baby_detection_simple_timeout.py
"""
車輛安全監控系統 - 極簡版（修正版）
功能：偵測到有人 + 熄火超過30秒 → 警報
"""

import sys
import time
import numpy as np
from PySide2 import QtWidgets, QtCore
from collections import deque

# ======== 簡單設定 ========
SETTING_FILE = r"C:\Users\user\Desktop\mmWave\radar-gesture-recognition-chore-update-20250815\TempParam\K60168-Test-00256-008-v0.0.8-20230717_60cm"

# 改進的偵測參數
MIN_ENERGY_THRESHOLD = 60.0  # 最低能量閾值：低於60就是沒人
DETECTION_THRESHOLD = 8.0  # 基礎能量閾值（相對於背景）
MOTION_THRESHOLD = 2.0  # 動作變化閾值
MIN_CONSECUTIVE_FRAMES = 5  # 連續偵測幀數（減少誤判）
ALERT_TIME = 30  # 熄火30秒後警報
# =======================

# KKT Module
try:
    from KKT_Module import kgl
    from KKT_Module.DataReceive.Core import Results
    from KKT_Module.DataReceive.DataReceiver import MultiResult4168BReceiver
    from KKT_Module.FiniteReceiverMachine import FRM
    from KKT_Module.SettingProcess.SettingConfig import SettingConfigs
    from KKT_Module.SettingProcess.SettingProccess import SettingProc
    from KKT_Module.GuiUpdater.GuiUpdater import Updater

    KKT_AVAILABLE = True
except:
    KKT_AVAILABLE = False
    print("⚠️ KKT_Module 未安裝")


def connect_device():
    try:
        device = kgl.ksoclib.connectDevice()
        if device != 'Unknow':
            print(f"✓ 雷達已連接: {device}")
            return True
    except:
        pass
    return False


def run_setting_script(setting_name):
    try:
        ksp = SettingProc()
        cfg = SettingConfigs()
        cfg.Chip_ID = kgl.ksoclib.getChipID().split(' ')[0]
        cfg.Processes = [
            'Reset Device', 'Gen Process Script',
            'Gen Param Dict', 'Get Gesture Dict',
            'Set Script', 'Run SIC',
            'Phase Calibration', 'Modulation On'
        ]
        cfg.setScriptDir(setting_name)
        ksp.startUp(cfg)
        print("✓ 雷達設定完成")
        return True
    except Exception as e:
        print(f"設定錯誤: {e}")
        return False


# ========== 改進的偵測器 ==========
class ImprovedPersonDetector:
    """改進的人體偵測器 - 多特徵融合"""

    def __init__(self):
        self.energy_buffer = deque(maxlen=30)  # 能量緩衝
        self.frame_buffer = deque(maxlen=5)  # 幀緩衝用於計算變化
        self.detection_buffer = deque(maxlen=10)  # 偵測結果緩衝

        self.has_person = False
        self.energy_level = 0.0
        self.motion_level = 0.0
        self.confidence = 0.0

        # 自適應背景
        self.background_energy = 0.0
        self.background_updated = False
        self.frame_count = 0

    def calculate_features(self, frame: np.ndarray):
        """計算多個特徵"""
        try:
            if frame.shape != (2, 32, 32):
                return None

            # 計算幅度
            magnitude = np.sqrt(frame[0] ** 2 + frame[1] ** 2)

            # 特徵1: 中心區域能量（人體通常在中心）
            center = magnitude[12:20, 12:20]  # 取中心 8x8 區域
            center_energy = np.mean(center)

            # 特徵2: 最大能量點
            max_energy = np.max(magnitude)

            # 特徵3: 整體平均能量
            avg_energy = np.mean(magnitude)

            # 特徵4: 能量方差（人體會有更高的方差）
            energy_std = np.std(magnitude)

            # 特徵5: 高能量點數量
            high_energy_points = np.sum(magnitude > (avg_energy + energy_std))

            # 組合特徵（加權平均）
            combined_energy = (
                    max_energy * 0.3 +  # 最大值權重
                    center_energy * 0.4 +  # 中心區域權重最高
                    avg_energy * 0.2 +  # 平均值
                    energy_std * 0.1  # 變化程度
            )

            return {
                'combined': combined_energy,
                'max': max_energy,
                'center': center_energy,
                'avg': avg_energy,
                'std': energy_std,
                'points': high_energy_points,
                'magnitude': magnitude
            }

        except Exception as e:
            print(f"特徵計算錯誤: {e}")
            return None

    def calculate_motion(self):
        """計算幀間變化（動作偵測）"""
        if len(self.frame_buffer) < 2:
            return 0.0

        try:
            current = self.frame_buffer[-1]
            previous = self.frame_buffer[-2]

            # 計算差異
            diff = np.abs(current - previous)
            motion = np.mean(diff)

            return motion
        except:
            return 0.0

    def update_background(self, energy):
        """自適應背景更新"""
        if not self.background_updated:
            # 初始化背景（前30幀的平均）
            if len(self.energy_buffer) >= 30:
                self.background_energy = np.percentile(list(self.energy_buffer), 25)  # 取25%分位數
                self.background_updated = True
                print(f"✓ 背景能量校準完成: {self.background_energy:.2f}")
        else:
            # 緩慢更新背景（只在無人時）
            if not self.has_person:
                self.background_energy = self.background_energy * 0.99 + energy * 0.01

    def push_frame(self, frame: np.ndarray):
        """主要偵測邏輯"""
        features = self.calculate_features(frame)
        if features is None:
            return False

        self.frame_count += 1

        # 儲存能量和幀數據
        energy = features['combined']
        self.energy_buffer.append(energy)
        self.frame_buffer.append(features['magnitude'])

        # 更新背景
        self.update_background(energy)

        # 計算動作
        motion = self.calculate_motion()
        self.motion_level = motion

        # ===== 第一層判斷：絕對能量閾值 =====
        # 如果能量低於 60，直接判定為沒人
        if energy < MIN_ENERGY_THRESHOLD:
            self.detection_buffer.append(False)
            self.has_person = False
            self.confidence = 0.0

            # 更新顯示用的能量值
            if len(self.energy_buffer) >= 5:
                self.energy_level = np.mean(list(self.energy_buffer)[-5:])
            else:
                self.energy_level = energy

            return False

        # 如果背景還沒校準完成，先不判斷
        if not self.background_updated:
            return False

        # ===== 第二層判斷：多條件分析 =====

        # 條件1: 能量超過自適應閾值
        energy_threshold = self.background_energy + DETECTION_THRESHOLD
        condition1 = energy > energy_threshold

        # 條件2: 中心區域有明顯信號
        condition2 = features['center'] > (self.background_energy + 5.0)

        # 條件3: 有動作或高能量點
        condition3 = (motion > MOTION_THRESHOLD) or (features['points'] > 10)

        # 條件4: 最大值夠高
        condition4 = features['max'] > (self.background_energy + 8.0)

        # 綜合判斷（至少滿足2個條件）
        score = sum([condition1, condition2, condition3, condition4])
        is_detected = score >= 2

        # 儲存偵測結果
        self.detection_buffer.append(is_detected)

        # 需要連續多幀偵測才確認（減少誤判）
        if len(self.detection_buffer) >= MIN_CONSECUTIVE_FRAMES:
            recent_detections = list(self.detection_buffer)[-MIN_CONSECUTIVE_FRAMES:]
            detection_ratio = sum(recent_detections) / MIN_CONSECUTIVE_FRAMES

            # 超過60%的幀都偵測到才確認有人
            self.has_person = detection_ratio >= 0.6
            self.confidence = detection_ratio * 100

        # 更新顯示用的能量值（使用移動平均平滑）
        if len(self.energy_buffer) >= 5:
            self.energy_level = np.mean(list(self.energy_buffer)[-5:])
        else:
            self.energy_level = energy

        # Debug輸出（每30幀）
        if self.frame_count % 30 == 0:
            status = "有人" if self.has_person else "無人"
            print(f"[偵測] 能量:{energy:.1f} | 背景:{self.background_energy:.1f} | "
                  f"動作:{motion:.1f} | 信心度:{self.confidence:.0f}% | "
                  f"狀態:{status}")

        return self.has_person


# ========== GUI 部分 ==========
class SimpleTimeoutGUI(QtWidgets.QMainWindow):
    update_signal = QtCore.Signal(bool, float, float, float)  # has_person, energy, motion, confidence

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 車輛安全監控系統 - 改進版")
        self.setGeometry(200, 200, 700, 650)
        self.setStyleSheet("background-color: #2C3E50;")

        self.engine_on = True
        self.has_person = False
        self.timer_count = 0
        self.alert_active = False

        self.timer_obj = QtCore.QTimer()
        self.timer_obj.timeout.connect(self.update_timer)

        self.update_signal.connect(self.update_detection_slot)

        self.setup_ui()

    def setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # 標題
        title = QtWidgets.QLabel("🚗 車輛安全監控系統")
        title.setStyleSheet("""
            font-size: 26px; font-weight: bold; color: white;
            background-color: #34495E; padding: 20px; border-radius: 10px;
        """)
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # 主要顯示區
        display_frame = QtWidgets.QFrame()
        display_frame.setStyleSheet("""
            background-color: #34495E; border-radius: 10px; padding: 30px;
        """)
        display_layout = QtWidgets.QVBoxLayout(display_frame)

        # 偵測狀態 - 超大圖示
        self.status_icon = QtWidgets.QLabel("⭕")
        self.status_icon.setStyleSheet("font-size: 120px; color: #95A5A6;")
        self.status_icon.setAlignment(QtCore.Qt.AlignCenter)
        display_layout.addWidget(self.status_icon)

        self.status_text = QtWidgets.QLabel("無人")
        self.status_text.setStyleSheet("font-size: 28px; color: white; font-weight: bold;")
        self.status_text.setAlignment(QtCore.Qt.AlignCenter)
        display_layout.addWidget(self.status_text)

        # 詳細資訊
        self.energy_text = QtWidgets.QLabel("能量: -- | 動作: -- | 信心度: --%")
        self.energy_text.setStyleSheet("font-size: 13px; color: #BDC3C7;")
        self.energy_text.setAlignment(QtCore.Qt.AlignCenter)
        display_layout.addWidget(self.energy_text)

        layout.addWidget(display_frame)

        # 系統資訊
        info_frame = QtWidgets.QFrame()
        info_frame.setStyleSheet("""
            background-color: #34495E; border-radius: 10px; padding: 20px;
        """)
        info_layout = QtWidgets.QVBoxLayout(info_frame)

        self.radar_label = QtWidgets.QLabel("📡 雷達: 初始化中...")
        self.radar_label.setStyleSheet("color: #F39C12; font-size: 14px;")
        info_layout.addWidget(self.radar_label)

        self.engine_label = QtWidgets.QLabel("🔑 引擎: 啟動中")
        self.engine_label.setStyleSheet("color: #2ECC71; font-size: 14px;")
        info_layout.addWidget(self.engine_label)

        self.timer_label = QtWidgets.QLabel("⏱ 熄火時間: 0 秒")
        self.timer_label.setStyleSheet("color: white; font-size: 14px;")
        info_layout.addWidget(self.timer_label)

        # 警報設定顯示
        self.setting_label = QtWidgets.QLabel(
            f"⚙️ 警報設定: 能量>{MIN_ENERGY_THRESHOLD:.0f} 且 熄火{ALERT_TIME}秒後觸發"
        )
        self.setting_label.setStyleSheet("color: #3498DB; font-size: 13px; font-weight: bold;")
        info_layout.addWidget(self.setting_label)

        layout.addWidget(info_frame)

        # 警報
        self.alert_label = QtWidgets.QLabel("⚠️ 警報！有人留在車內超過30秒！")
        self.alert_label.setStyleSheet("""
            background-color: #E74C3C; color: white;
            font-size: 20px; font-weight: bold;
            padding: 20px; border-radius: 10px;
        """)
        self.alert_label.setAlignment(QtCore.Qt.AlignCenter)
        self.alert_label.hide()
        layout.addWidget(self.alert_label)

        # 控制按鈕
        self.engine_btn = QtWidgets.QPushButton("🔑 熄火")
        self.engine_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px; font-weight: bold; color: white;
                background-color: #E74C3C; padding: 18px;
                border-radius: 10px;
            }
            QPushButton:hover { background-color: #C0392B; }
        """)
        self.engine_btn.clicked.connect(self.toggle_engine)
        layout.addWidget(self.engine_btn)

    def toggle_engine(self):
        self.engine_on = not self.engine_on

        if self.engine_on:
            self.engine_btn.setText("🔑 熄火")
            self.engine_btn.setStyleSheet("""
                QPushButton {
                    font-size: 18px; font-weight: bold; color: white;
                    background-color: #E74C3C; padding: 18px;
                    border-radius: 10px;
                }
                QPushButton:hover { background-color: #C0392B; }
            """)
            self.engine_label.setText("🔑 引擎: 啟動中")
            self.engine_label.setStyleSheet("color: #2ECC71; font-size: 14px;")
            self.timer_count = 0
            self.timer_obj.stop()
        else:
            self.engine_btn.setText("🔑 啟動")
            self.engine_btn.setStyleSheet("""
                QPushButton {
                    font-size: 18px; font-weight: bold; color: white;
                    background-color: #2ECC71; padding: 18px;
                    border-radius: 10px;
                }
                QPushButton:hover { background-color: #27AE60; }
            """)
            self.engine_label.setText("🔑 引擎: 已熄火")
            self.engine_label.setStyleSheet("color: #E74C3C; font-size: 14px;")
            self.timer_obj.start(1000)

        self.check_alert()

    def update_timer(self):
        self.timer_count += 1
        self.timer_label.setText(f"⏱ 熄火時間: {self.timer_count} 秒")

        # 顯示倒數提示
        if self.has_person and not self.engine_on:
            remaining = ALERT_TIME - self.timer_count
            if remaining > 0:
                self.timer_label.setText(f"⏱ 熄火時間: {self.timer_count} 秒 (警報倒數: {remaining}秒)")

        self.check_alert()

    def update_detection(self, has_person, energy, motion, confidence):
        self.update_signal.emit(has_person, energy, motion, confidence)

    @QtCore.Slot(bool, float, float, float)
    def update_detection_slot(self, has_person, energy, motion, confidence):
        self.has_person = has_person

        if has_person:
            self.status_icon.setText("👤")
            self.status_icon.setStyleSheet("font-size: 120px; color: #E74C3C;")
            self.status_text.setText("偵測到人！")
            self.status_text.setStyleSheet("font-size: 28px; color: #E74C3C; font-weight: bold;")
        else:
            self.status_icon.setText("⭕")
            self.status_icon.setStyleSheet("font-size: 120px; color: #95A5A6;")
            self.status_text.setText("無人")
            self.status_text.setStyleSheet("font-size: 28px; color: white; font-weight: bold;")

        # 顯示能量狀態
        energy_status = f"能量: {energy:.1f}"
        if energy < MIN_ENERGY_THRESHOLD:
            energy_status += " (太低)"

        self.energy_text.setText(f"{energy_status} | 動作: {motion:.1f} | 信心度: {confidence:.0f}%")

        self.check_alert()

    def check_alert(self):
        """簡單警報邏輯"""
        # 條件: 熄火 + 有人 + 超過30秒
        should_alert = (not self.engine_on) and self.has_person and (self.timer_count >= ALERT_TIME)

        if should_alert and not self.alert_active:
            self.alert_label.show()
            self.alert_active = True
            print(f"🚨 警報觸發! 熄火 {self.timer_count} 秒")
        elif not should_alert and self.alert_active:
            self.alert_label.hide()
            self.alert_active = False

    def update_radar_status(self, connected):
        if connected:
            self.radar_label.setText("📡 雷達: 已連接 ✓ (校準中...)")
            self.radar_label.setStyleSheet("color: #F39C12; font-size: 14px;")
        else:
            self.radar_label.setText("📡 雷達: 未連接")
            self.radar_label.setStyleSheet("color: #E74C3C; font-size: 14px;")

    def update_radar_calibrated(self):
        self.radar_label.setText("📡 雷達: 已連接 ✓ (已校準)")
        self.radar_label.setStyleSheet("color: #2ECC71; font-size: 14px;")


# 簡單更新器（修正版）
class SimpleTimeoutUpdater(Updater):
    def __init__(self, detector, gui):
        super().__init__()
        self.detector = detector
        self.gui = gui
        self.frame_count = 0
        self.calibration_notified = False

    def update(self, res: Results):
        try:
            # 檢查資料是否有效
            if not hasattr(res, 'feature_map') or res['feature_map'] is None:
                return

            arr = res['feature_map'].data
            if arr is None:
                return

            frame = self._to_frame(arr)
            if frame is None:
                return

            has_person = self.detector.push_frame(frame)

            # 校準完成通知
            if self.detector.background_updated and not self.calibration_notified:
                self.gui.update_radar_calibrated()
                self.calibration_notified = True

            # 每10幀更新一次GUI
            if self.frame_count % 10 == 0:
                self.gui.update_detection(
                    has_person,
                    self.detector.energy_level,
                    self.detector.motion_level,
                    self.detector.confidence
                )

            self.frame_count += 1

        except KeyError as e:
            if self.frame_count % 100 == 0:
                print(f"資料鍵錯誤: {e}")
        except AttributeError as e:
            if self.frame_count % 100 == 0:
                print(f"屬性錯誤: {e}")
        except Exception as e:
            if self.frame_count % 100 == 0:
                print(f"更新錯誤: {e}")

    @staticmethod
    def _to_frame(arr):
        try:
            x = np.asarray(arr, dtype=np.float32)

            # 檢查並轉換格式
            if x.shape == (2, 32, 32):
                return x
            elif x.shape == (32, 32, 2):
                return np.transpose(x, (2, 0, 1))
            elif x.size == 2048:  # 2 * 32 * 32
                return x.reshape(2, 32, 32)
            else:
                print(f"未知的資料形狀: {x.shape}")
                return None

        except Exception as e:
            print(f"幀轉換錯誤: {e}")
            return None


# 主程式
def main():
    app = QtWidgets.QApplication(sys.argv)

    gui = SimpleTimeoutGUI()
    gui.show()

    if not KKT_AVAILABLE:
        gui.update_radar_status(False)
        print("💡 KKT_Module 未安裝，使用模擬模式")
        return app.exec_()

    detector = None  # 先初始化為 None

    try:
        print("正在連接雷達...")
        kgl.setLib()

        if not connect_device():
            gui.update_radar_status(False)
            return app.exec_()

        print("正在設定雷達...")
        if not run_setting_script(SETTING_FILE):
            gui.update_radar_status(False)
            return app.exec_()

        kgl.ksoclib.writeReg(1, 0x50000504, 5, 5, 0)

        gui.update_radar_status(True)
        print("✓ 雷達初始化完成")
        print(f"⚙️ 警報設定: 能量 > {MIN_ENERGY_THRESHOLD:.0f} 且熄火 {ALERT_TIME} 秒後觸發")
        print("⏳ 背景校準中（需要約30幀）...")

        # 在這裡創建 detector
        detector = ImprovedPersonDetector()
        updater = SimpleTimeoutUpdater(detector, gui)

        receiver = MultiResult4168BReceiver()
        receiver.actions = 1
        receiver.rbank_ch_enable = 7
        receiver.read_interrupt = 0
        receiver.clear_interrupt = 0

        FRM.setReceiver(receiver)
        FRM.setUpdater(updater)
        FRM.trigger()
        FRM.start()

        print("✓ 系統啟動完成")
        print("💡 使用說明:")
        print("   1. 系統會自動校準背景環境")
        print(f"   2. 能量低於 {MIN_ENERGY_THRESHOLD:.0f} 會直接判定為無人")
        print("   3. 在雷達前方有人會顯示 👤")
        print("   4. 點擊「熄火」按鈕")
        print(f"   5. 等待 {ALERT_TIME} 秒後警報觸發")

    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        import traceback
        traceback.print_exc()
        gui.update_radar_status(False)

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\n正在關閉...")
    finally:
        try:
            FRM.stop()
            kgl.ksoclib.closeDevice()
            print("✓ 雷達已關閉")
        except:
            pass


if __name__ == "__main__":
    main()