package riemannzeta;

public class PrimeList {
	public static final long[] primeList = primesUpTo(65536);
	public static long[] primesUpTo(int n) {
		long[] integers = new long[n];
		long k = 2;
		int p = 0;
		while (p < n) {
			if (isPrime(k)) {
				integers[p] = k;
				p++;
			}
			k++;
		}
		return integers;
	}
	
	public static boolean isPrime(long n) {
		if (n < 2)
			return false;
		if (n == 2 || n == 3)
			return true;
		for (int k = 2; k*k <= n; k++) {
			if (n % k == 0)
				return false;
		}
		return true;
	}
	
	public static void main(String[] args) {
		final int MULT = 256;
		final int N = 256;
		int n = MULT;
		
		System.out.println("n\tp\tms\t");
		for (int i = 1; i <= N; i++) {
			int time = (int) System.currentTimeMillis();
			final long[] P = primesUpTo(n);
			time = (int) System.currentTimeMillis() - time;
			
			System.out.printf("%d\t%d\t%d\n", 
					i, P[P.length - 1], time);
			
			n += MULT;
		}
	}
}
