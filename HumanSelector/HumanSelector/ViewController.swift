//
//  ViewController.swift
//  HumanSelector
//
//  Created by Taro Kimura on 2018/02/10.
//  Copyright © 2018年 Taro Kimura. All rights reserved.
//

import UIKit
import CoreML
import Vision
import ImageIO


class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    @IBOutlet weak var cameraView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!
    
    var inputImage: CIImage!
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    func imagePickerController(_ imagePicker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        self.resultLabel.text = "Analyzing Image…"
        if let pickedImage = info[UIImagePickerControllerOriginalImage] as? UIImage {
            self.cameraView.contentMode = .scaleAspectFit
            self.cameraView.image = pickedImage
        }
        imagePicker.dismiss(animated: true, completion: {
            guard let uiImage = info[UIImagePickerControllerOriginalImage] as? UIImage
                else { fatalError("no image from image picker") }
            guard let ciImage = CIImage(image: uiImage)
                else { fatalError("can't create CIImage from UIImage") }
            let orientation = CGImagePropertyOrientation(rawValue: UInt32(uiImage.imageOrientation.rawValue))
            self.inputImage = ciImage.oriented(forExifOrientation: Int32(orientation!.rawValue))
            
            //リクエストハンドラの作成。ここでカメラで撮影した画像を渡します。
            let handler = VNImageRequestHandler(ciImage: self.inputImage)
            do {
                try handler.perform([self.classificationRequest_vgg])
            } catch {
                print(error)
            }
        })
    }
    
    //VGG16学習モデルで画像を認識させるリクエストの作成
    lazy var classificationRequest_vgg: VNCoreMLRequest = {
        do {
            var model: VNCoreMLModel? = nil
            model = try VNCoreMLModel(for: peopleornot2().model)
            return VNCoreMLRequest(model: model!, completionHandler: self.handleClassification)
        } catch {
            fatalError("can't load Vision ML model: \(error)")
        }
    }()
    
    //学習モデルが持っているラベルに応じて分類した結果の表示。分類結果の１個目（一番confidenceが高いもの）を表示します。
    //分類結果はVNClassificationObservation型のオブジェクトで返ってきます。
    func handleClassification(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNClassificationObservation]
            else { fatalError("unexpected result type from VNCoreMLRequest") }
        guard let best = observations.first
            else { fatalError("can't get best result") }
        
        DispatchQueue.main.async {
            let classification: String = (best.identifier);
            self.resultLabel.text = "Classification: \"\(classification)\" Confidence: \(best.confidence)"
        }
    }
    
    @IBAction func openCamera(_ sender: Any) {
        let sourceType:UIImagePickerControllerSourceType = UIImagePickerControllerSourceType.camera
        if UIImagePickerController.isSourceTypeAvailable(UIImagePickerControllerSourceType.camera){
            let cameraPicker = UIImagePickerController()
            cameraPicker.sourceType = sourceType
            cameraPicker.delegate = self
            self.present(cameraPicker, animated: true, completion: nil)
        } else {
            print("error")
        }
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
}
