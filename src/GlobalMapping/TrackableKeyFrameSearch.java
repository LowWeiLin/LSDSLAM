package GlobalMapping;

public class TrackableKeyFrameSearch {
	
	static float KFDistWeight = 4;
	static float KFUsageWeight = 3;
	
	public float getRefFrameScore(float distanceSquared, float usage) {
		return distanceSquared*KFDistWeight*KFDistWeight
				+ (1-usage)*(1-usage) * KFUsageWeight * KFUsageWeight;
	}
}
