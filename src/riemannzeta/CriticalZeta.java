package riemannzeta;

import java.util.Scanner;

import complex.Complex;
import riemannzeta.PrimeList;

import java.util.concurrent.CyclicBarrier;

@SuppressWarnings("unused")
public class CriticalZeta {
	private static final int THREADS = 12;
	private static final Complex[] sums = new Complex[THREADS];
	private static final CyclicBarrier barrier = new CyclicBarrier(THREADS);
	public static float zeta(double s, int terms) {
		if (s <= 1)
			return Float.POSITIVE_INFINITY;
		double sum = 0;
		double term = 0;
		int n = 1;
		do {
			term = 1/Math.pow(n, s);
			sum += term;
			n++;
		} while (n <= terms);
		return (float) sum;
	}
	
	public static float zetaPrime(double s) {
		if (s <= 1)
			return Float.POSITIVE_INFINITY;
		double product = 1;
		double term = 0;
		int n = 1;
		do {
			long p = PrimeList.primeList[n - 1];
			term = 1/(1-Math.pow(p, -s));
			product *= term;
			n++;
		} while (Math.abs(1 - term) >= 1E-09);
		return (float) product;
	}
	
	public static float eta(double s, int terms) {
		if (s <= 0)
			return Float.POSITIVE_INFINITY;
		double sum = 0;
		int n = 1;
		while (n < terms) {
			int sgn = (n % 2 != 0 ? 1 : -1);
			sum += sgn/Math.pow(n, s);
			n++;
		}
		return (float) sum;
	}
	
	public static float zetaFromEta(double s, int terms) {
		return (float) (eta(s, terms)/(1 - Math.pow(2, 1-s)));
	}
	
	public static Complex eta(Complex s, int terms) {
		/* Not implemented on the left halfplane */
		if (s.re() <= 0)
			return new Complex(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY);
		if (s.re() == 1 && s.im() == 1)
			return new Complex(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY);
		Complex sum = new Complex(0,0);
		Complex sum2 = sum;
		int n = 1;
		/* Can be performed in parallel. Round up. */
		Thread[] threads = new Thread[THREADS];
		final int SUBTERMS = (THREADS + terms - 1)/THREADS;
		final int TOTALTERMS = SUBTERMS*THREADS;
		int ini = 0;
		for (int t = 0; t < THREADS; t++) {
			Thread thread = new Thread(new TermThread(t, s, ini + 1, ini + SUBTERMS));
			threads[t] = thread;
			thread.start();
			ini += SUBTERMS;
		}
		for (Thread thread : threads) {
			try {
				thread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		// Average out sums with positive and negative trailing term. Must account for rounding up.
		for (Complex subsum : sums) {
			sum = sum.add(subsum);
		}
		sum2 = sum.add(etaTerm(s, TOTALTERMS + 1));
		return sum.add(sum2).div(2);
	}
	
	private static class TermThread implements Runnable {
		private int t, ini, fin;
		private Complex s, sum;
		public TermThread(int t, Complex s, int ini, int fin) {
			this.t = t;
			this.ini = ini;
			this.fin = fin;
			this.s = s;
			this.sum = new Complex(0,0);
		}
	    public void run()
	    {
	        try {
	        	// System.out.printf("^[%d]: %f, %d, %d\n", t, s.im(), ini, fin);
	            for (int k = ini; k <= fin; k++) {
	            	sum = sum.add(etaTerm(s,k));
	            }
	            sums[t] = sum;
	            barrier.await();
	            // System.out.printf("\t$[%d]: %f, %d, %d\n", t, s.im(), ini, fin);
	        }
	        catch (Exception e) {
	            e.printStackTrace();
	        }
	    }
	}
	
	public static Complex etaTerm(Complex s, int n) {
		Complex term = new Complex((double) 1/n,0);
		int sgn = n % 2 != 0 ? 1 : -1;
		term = term.raise(s);
		term = term.mult(new Complex(sgn,0));
		return term;
	}
	
	public static Complex zeta(Complex s, int terms) {
		if (s.re() <= 0)
			return new Complex(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY);
		if (s.re() == 1 && s.im() == 1)
			return new Complex(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY);
		Complex t = new Complex(1,0).sub(s);
		Complex divisor = (new Complex(1,0).sub(new Complex(2,0).raise(t)));
		return eta(s, terms).div(divisor);
	}
	
	public static Complex zetaTerm(Complex s, int n) {
		Complex t = new Complex(1,0).sub(s); // ^1-s
		Complex divisor = (new Complex(1,0).sub(new Complex(2,0).raise(t)));
		return etaTerm(s,n).div(divisor);
	}
	/*
	public static void main(String[] args) {
		boolean finish = false;
		while(!finish) {
			Scanner key = new Scanner(System.in);
			System.out.print("Enter Re part: ");
			double re = key.nextDouble();
			if (re == -1)
				finish = true;
			System.out.print("Enter Im part: ");
			double im = key.nextDouble();
			Complex s = new Complex(re,im);
			System.out.printf("zeta(%f,%f) = %s\n", re, im, zeta(s,(int)10E6));
			key.close();
		}
	}
	*/
}
