import java.io.File;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;

import main.Sim3_graph_libraryLibrary;

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
			
			// TODO: Do optimization here?
			// Do after constraint search, if new KF is added.
			//lsdSlam.optimizationIteration(50, 0.001);
			
			// WRITE POINT CLOUD TO FILE
			if (lsdSlam.flushPC == true) {
				try {
					//keyFrameGraph.writePointCloudToFile("graphPOINTCLOUD-" + currentKeyFrame.id() + ".ply");
					lsdSlam.keyFrameGraphDisplay.writePointCloudToFile("graphPOINTCLOUD-" + (imageSeqNumber-1) + ".ply");
				} catch (FileNotFoundException | UnsupportedEncodingException e) {
				}
				lsdSlam.flushPC = false;
			}
			
		}
		
	}
	

	public static void main(String[] args) {
		// Load OpenCV native library.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		Sim3_graph_libraryLibrary.clear();
		
		// Set Camera parameters
		
		// LSDSLAM example
		Constants.setK(254.327, 375.934, 266.882, 231.099);
		
		// GoPro - studio undistort
		//Constants.setK(291.55136695, 391.96175257, 316.61752775, 246.73475848);
		
		
		SLAMWrapper liveSlamWrapper = new SLAMWrapper();
		liveSlamWrapper.loop();
		
	}
	
}
