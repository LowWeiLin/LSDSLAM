package Utils;

public class Utils {
	
	public static float interpolatedValue(byte[] dataArray, double x, double y, int width) {

		int ix = (int)x;
		int iy = (int)y;
		float dx = (float) (x - ix);
		float dy = (float) (y - iy);
		float dxdy = dx*dy;
		int bp = ix+iy*width;
		
		float res =   dxdy 			* (float)((int)dataArray[bp+1+width])
					+ (dy-dxdy) 	* (float)((int)dataArray[bp+width])
					+ (dx-dxdy) 	* (float)((int)dataArray[bp+1])
					+ (1-dx-dy+dxdy)* (float)((int)dataArray[bp]);

		return res;
		
	}
	
	public static float interpolatedValue(float[] dataArray, double x, double y, int width) {

		int ix = (int)x;
		int iy = (int)y;
		float dx = (float) (x - ix);
		float dy = (float) (y - iy);
		float dxdy = dx*dy;
		int bp = ix+iy*width;
		
		float res =   dxdy 			* dataArray[bp+1+width]
					+ (dy-dxdy) 	* dataArray[bp+width]
					+ (dx-dxdy) 	* dataArray[bp+1]
					+ (1-dx-dy+dxdy)* dataArray[bp];
		//System.out.println("interpolatedValue: " + dataArray[bp]);
		return res;
		
	}
}
