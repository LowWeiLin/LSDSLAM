package DataStructures;
import org.opencv.core.Mat;


public class Frame {

	public Mat image;
	public byte[] imageArray; // For fast reading
	
	public Frame(Mat image) {
		this.image = image;
		this.imageArray = new byte[(int) image.total()];
		this.image.get(0, 0, imageArray);
	}
	
	public void getGradient() {
		
	}
	
	public int width() {
		return image.width();
	}
	
	public int height() {
		return image.height();
	}
	
}
