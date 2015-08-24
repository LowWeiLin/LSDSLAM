import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

import Utils.Constants;


public class LiveSLAMWrapper {
	
	boolean isInitialized = false;
	int imageSeqNumber = 0;
	LSDSLAM lsdSlam = new LSDSLAM();
	
	public LiveSLAMWrapper() {
	}
	
	public void loop() {
		
		VideoCapture capture = new VideoCapture();
		capture.open("table-vid-trimmed.avi");
		
		while(true) {
			Mat frame = new Mat();
			boolean success = capture.read(frame);
			
			if (!success) {
				break;
			}
			
			// Process image frame
			newImageCallback(frame);
		}
		
		
		// Read image
//		Mat image1 = null;
//		Mat image2 = null;
//		image1 = Highgui.imread("test1.jpg");
//		image2 = Highgui.imread("test2.jpg");
//		newImageCallback(image1);
//		newImageCallback(image2);
//		newImageCallback(image1);
//		newImageCallback(image2);
		
		
		
	}
	
	public void newImageCallback(Mat image) {
		System.out.println("-------\nFrame " + imageSeqNumber + "\n-------");
		
		// Increment image sequence number
		imageSeqNumber++;

		// Convert image to grayscale, if necessary
		Mat grayImg;
		if (image.channels() == 1) { 
			grayImg = image;
		} else {
			grayImg = new Mat();
			Imgproc.cvtColor(image, grayImg, Imgproc.COLOR_RGB2GRAY);
		}

		// Assert that we work with 8 bit images
		assert(grayImg.elemSize() == 1);

		// need to initialize
		if(!isInitialized) {
			lsdSlam.randomInit(grayImg, 1);
			isInitialized = true;
		} else if(isInitialized && lsdSlam != null) {
			lsdSlam.trackFrame(grayImg, imageSeqNumber);
			

			
			// TODO: remove? call in another thread?
			// Do here, sequentially for now.
			lsdSlam.doMappingIteration();
			
		}
		
	}
	

	public static void main(String[] args) {
		// Load OpenCV native library.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// Set Camera parameters
		Constants.setK(748, 748, 319, 239);
				
		LiveSLAMWrapper liveSlamWrapper = new LiveSLAMWrapper();
		liveSlamWrapper.loop();
		
	}
	
}
