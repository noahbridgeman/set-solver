// Kamera -> Erkennung (SetCore) -> veröffentlicht Bild + Ergebnis für die Anzeige.
// Läuft im Hochformat; das Kamerabild wird dafür um 90° gedreht geliefert (1080 x 1920).
import AVFoundation
import CoreImage
import QuartzCore
import SetCore
import SwiftUI

final class CameraModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var frame: CGImage?
    @Published var result: FrameResult?
    @Published var fps: Double = 0
    @Published var ultraWide = false
    @Published var hasUltraWide = false
    @Published var paused = false
    @Published var message: String?

    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "kamera", qos: .userInitiated)
    private let output = AVCaptureVideoDataOutput()
    private var input: AVCaptureDeviceInput?
    private var pipeline: LivePipeline?
    private let displayContext = CIContext()
    private var lastTime = CACurrentMediaTime()
    private var configured = false
    private var isPaused = false  // Kopie für den Kamera-Thread

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { ok in
            guard ok else {
                DispatchQueue.main.async { self.message = "Kein Kamerazugriff. Bitte in Einstellungen → SET erlauben." }
                return
            }
            self.queue.async {
                if !self.configured { self.configure() }
                if !self.session.isRunning { self.session.startRunning() }
            }
        }
    }

    func stop() { queue.async { self.session.stopRunning() } }

    func setUltraWide(_ on: Bool) {
        queue.async {
            self.session.beginConfiguration()
            self.selectCamera(ultraWide: on)
            self.session.commitConfiguration()
            self.applyRotation()
            self.pipeline?.tracker.reset()
        }
    }

    func setPaused(_ p: Bool) {
        paused = p
        queue.async { self.isPaused = p }
    }

    func resetTracking() { queue.async { self.pipeline?.tracker.reset() } }

    // MARK: - Einrichtung (Kamera-Thread)

    private func configure() {
        do {
            guard let det = Bundle.main.url(forResource: "SetDetector", withExtension: "mlmodelc"),
                  let cls = Bundle.main.url(forResource: "SetClassifier", withExtension: "mlmodelc") else {
                throw NSError(domain: "SET", code: 1, userInfo: [NSLocalizedDescriptionKey: "Modelle fehlen in der App"])
            }
            pipeline = try LivePipeline(detectorURL: det, classifierURL: cls)
        } catch {
            DispatchQueue.main.async { self.message = "Modelle konnten nicht geladen werden: \(error.localizedDescription)" }
            return
        }
        let ultra = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back) != nil
        session.beginConfiguration()
        session.sessionPreset = .hd1920x1080
        selectCamera(ultraWide: false)
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true  // lieber Bilder auslassen als hinterherhinken
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()
        applyRotation()
        configured = true
        DispatchQueue.main.async { self.hasUltraWide = ultra }
    }

    private func selectCamera(ultraWide: Bool) {
        let type: AVCaptureDevice.DeviceType = ultraWide ? .builtInUltraWideCamera : .builtInWideAngleCamera
        guard let device = AVCaptureDevice.default(type, for: .video, position: .back),
              let newInput = try? AVCaptureDeviceInput(device: device) else { return }
        if let old = input { session.removeInput(old) }
        if session.canAddInput(newInput) {
            session.addInput(newInput)
            input = newInput
        } else if let old = input {
            session.addInput(old)
            return
        }
        if (try? device.lockForConfiguration()) != nil {
            if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
            if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
            device.unlockForConfiguration()
        }
        DispatchQueue.main.async { self.ultraWide = ultraWide }
    }

    private func applyRotation() {
        guard let c = output.connection(with: .video) else { return }
        if c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }  // Hochformat
    }

    // MARK: - Pro Kamerabild (Kamera-Thread)

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard !isPaused, let pipeline, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let image = CIImage(cvPixelBuffer: pb)
        let res: FrameResult
        do {
            res = try pipeline.process(image)
        } catch {
            return
        }
        let cg = displayContext.createCGImage(image, from: image.extent)
        let now = CACurrentMediaTime()
        let inst = 1 / max(now - lastTime, 1e-3)
        lastTime = now
        DispatchQueue.main.async {
            self.frame = cg
            self.result = res
            self.fps = self.fps == 0 ? inst : 0.9 * self.fps + 0.1 * inst
        }
    }
}
