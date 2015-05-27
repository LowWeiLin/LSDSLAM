import java.awt.EventQueue;

import org.opencv.core.Core;

import IO.VideoCapBuffer;
import UI.DisplayJFrame;

/**
 * Entry point
 */
public class Main {
	public static void main(String[] args) {
		
		// Say Hello
		System.out.println("Hello world");
		
		// Load OpenCV native library.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// Start capturing video
		final VideoCapBuffer capBuffer = new VideoCapBuffer();
		
		// Start JFrame to display raw frame data
		EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    DisplayJFrame frame = new DisplayJFrame();
                    frame.frameBuffer = capBuffer.frameRGBBuffer;
                    frame.setVisible(true);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
		
		
		// Start LSDSLAM
		// =============
		
		
		
		
	}
}
