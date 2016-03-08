import java.io.File;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import Utils.Constants;


public class SLAMWrapper {
	
	boolean isInitialized = false;
	int imageSeqNumber = 0;
	LSDSLAM lsdSlam = new LSDSLAM();
	
	final String imagesDirectory = "images";
	
	// g_2 has lots of specular reflections/highlights.
	
	public SLAMWrapper() {
	}
	
	public void loop() {
		
		// Use images in images directory
		String currentDir = System.getProperty("user.dir");
		final File folder = new File(currentDir + "/" + imagesDirectory);
		
		for (final File fileEntry : folder.listFiles()) {
	        if (fileEntry.isDirectory()) {
	        } else {
	        	System.out.println("Reading: " + imagesDirectory + "/" + fileEntry.getName());
	            Mat frame = Highgui.imread(imagesDirectory + "/" + fileEntry.getName());
	            
	            if (frame.empty()) {
	            	System.out.println("Invalid image frame");
	            	continue;
	            }
	            
				// Process image frame
				newImageCallback(frame);
	        }
	    }
		
	}
	
	public void newImageCallback(Mat image) {
		System.gc();
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
			
			
			// TODO: Do constraint search here?
			lsdSlam.constraintSearchIteration();
			
		}
		
	}
	

	public static void main(String[] args) {
		// Load OpenCV native library.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// Set Camera parameters
		// My webcam
		//Constants.setK(748.000000, 748.000000, 319.000000, 239.000000);
		
		// LSDSLAM example
		Constants.setK(254.327, 375.934, 266.882, 231.099);
		
		
		// GoPro - studio undistort old
		//Constants.setK(293.44402418, 393.75304703, 319.60465583, 252.00499988);
		
		// GoPro - studio undistort
		//Constants.setK(291.55136695, 391.96175257, 316.61752775, 246.73475848);
		
		// GoPro - undistort twice
		//Constants.setK(282.3554769, 282.37401685, 316.8959948, 245.18105368);
		
		
		SLAMWrapper liveSlamWrapper = new SLAMWrapper();
		liveSlamWrapper.loop();
		
	}
	
}
