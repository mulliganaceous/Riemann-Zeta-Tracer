package mandelbrot;

import complex.Complex;

public class Mandelbrot {
	public static int Mandelbrot(Complex z0, int iterations) {
		Complex z = z0;
		int j = 0;
		while (z.abs() <= 2 && j <= iterations) {
			System.out.println("#" + j + ":\t" + z);
			z = z.mult(z).add(z0);
			
			if (j % 256 == 0)
				System.out.println("/");
			j++;
		}
		return (j > iterations? -1 : j);
	}
	
	public static Complex iterate(Complex z, Complex z0) {
		return z.mult(z).add(z0);
	}
	
	public static void main(String[] args) {
		final double re = -0.706656415693871;
		final double im = +0.236382660750009;
		Complex z = new Complex(re, im);
		System.out.println(Mandelbrot(z, 65536));
	}
}
