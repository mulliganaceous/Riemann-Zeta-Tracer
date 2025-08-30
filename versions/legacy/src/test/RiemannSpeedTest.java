package test;

import complex.Complex;
import riemannzeta.CriticalZeta;

public class RiemannSpeedTest {
	public static final int INTERVAL_LEVEL = 64;
	public static final double INTERVAL = 1d/INTERVAL_LEVEL;
	public static final int ACCURACY_LEVEL = 65536;
	public static long testRiemannSpeedR() {
		long start = System.currentTimeMillis();
		
		double s = INTERVAL;
		double z;
		while (s <= 16) {
			z = CriticalZeta.zetaFromEta(s, ACCURACY_LEVEL);
			System.out.printf("zeta(%f) = %f\n", s, z);
			s += INTERVAL;
		}
		
		return System.currentTimeMillis() - start;
	}
	
	public static long testRiemannSpeedC() {
		long start = System.currentTimeMillis();
		
		Complex s = new Complex(INTERVAL, 0);
		Complex z;
		while (s.re() <= 16) {
			z = CriticalZeta.zeta(s, ACCURACY_LEVEL);
			System.out.printf("zeta(%f) = %f\n", s.re(), z.re());
			s = s.add(new Complex(INTERVAL, 0));
		}
		
		return System.currentTimeMillis() - start;
	}
	
	public static void main(String[] args) {
		System.out.println(testRiemannSpeedR());
		System.out.println(testRiemannSpeedC());
	}
}
