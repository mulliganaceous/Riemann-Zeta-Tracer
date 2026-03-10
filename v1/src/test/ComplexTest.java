package test;

import complex.Complex;
import riemannzeta.CriticalZeta;

class ComplexTest {
	public static void TestCase1() {
		Complex[] z = new Complex[6];
		z[0] = new Complex(1,0);
		z[1] = new Complex(-1,3.14);
		z[2] = new Complex(2,-3.14);
		z[3] = new Complex(-Math.E, -Math.PI);
		z[4] = new Complex(Math.sqrt(2),Math.sqrt(2));
		z[5] = new Complex(0.01, 0.03);
		
		for (Complex zi : z) {
			System.out.printf("%-20s\t", zi.toStringDigits(3));
			System.out.printf("%.3f,%.2f\t", zi.abs(), Math.toDegrees(zi.arg()));
			System.out.println();
		}
		
		for (int i = 0; i < 6; i++) {
			for (int j = i; j < 6; j++) {
				String zStr = z[i].mult(z[j]).toStringDigits(3);
				System.out.print("z" + i + "*" + "z" + j + " = " + zStr + "\t");
				System.out.print(z[i].mult(z[j]).div(z[j]).toStringDigits(3) + "\t");
				System.out.print(z[j].mult(z[i]).div(z[i]).toStringDigits(3) + "\n");
			}
		}
	}
	
	public static void TestCase2() {
		double re1 = 1;
		double im1 = 0;
		double re2 = 10;
		double im2 = 0;
		Complex z1 = new Complex(re1, im1);
		Complex z2 = new Complex(re2, im2);
		System.out.println(z1.div(z2));
	}
	
	public static void TestCase3() {
		Complex z1 = new Complex(2,0);
		Complex z2 = new Complex(3,2);
		System.out.println(z1.raise(z2));
	}
	
	public static Complex zetaTermTest(Complex s, int terms) {
		if (s.re() <= 0)
			return new Complex(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY);
		if (s.re() == 1 && s.im() == 1)
			return new Complex(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY);
		
		Complex sum = new Complex(0,0);
		Complex sum2 = sum;
		final Complex t = new Complex(1,0).sub(s);
		int n = 1;
		while (n <= terms) {
			int sgn = (n % 2 != 0 ? 1 : -1);
			Complex term = new Complex((double) 1/n,0);
			Complex divisor = (new Complex(1,0).sub(new Complex(2,0).raise(t)));
			term = term.raise(s);
			term = term.div(divisor);
			term = term.mult(new Complex(sgn,0));
			
			System.out.println(term);
			if (n < terms)
				sum = sum.add(term);
			else
				sum2 = sum.add(term);
			n++;
		}
		return sum.add(sum2).div(new Complex(2,0));
	}
	
	public static Complex zetaPrintTerms(Complex s, int terms) {
		if (s.re() <= 0)
			return new Complex(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY);
		if (s.re() == 1 && s.im() == 1)
			return new Complex(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY);
		Complex sum = new Complex(0,0);
		Complex sum2 = sum;
		int n = 1;
		while (n <= terms) {
			Complex term = CriticalZeta.zetaTerm(s,n);
			System.out.println(term);
			sum = sum.add(term);
			n++;
		}
		sum2 = sum.add(CriticalZeta.zetaTerm(s,n));
		return sum.add(sum2).div(new Complex(2,0));
	}
	
	public static void main(String[] args) {
		System.out.println(zetaPrintTerms(new Complex(0.5,0),22));
		System.out.println(CriticalZeta.zeta(new Complex(0.5,0),65536));
	}
}
