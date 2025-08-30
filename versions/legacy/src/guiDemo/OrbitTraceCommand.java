package guiDemo;

import java.awt.Color;
import java.awt.Graphics;
import complex.Complex;
import mandelbrot.Mandelbrot;

public class OrbitTraceCommand {
	private MandelbrotTraceModel model;
	final static int ITERATIONS = MandelbrotTraceModel.getIterations();
	final int OO = MandelbrotTracePanel.ORIGIN;
	final int U = MandelbrotTracePanel.UNIT;
	
	public OrbitTraceCommand(MandelbrotTraceModel model) {
		this.model = model;
	}
	
	public void execute(Graphics g) {
		Complex z0 = model.getComplex();
		Complex z = z0;
		int j = 0;
		int x1 = (int) (OO + z.re()*U);
		int y1 = (int) (OO - z.im()*U);
		int x2, y2;
		// Iterative step
		while (z.abs() <= 2 && j <= ITERATIONS) {
			z = Mandelbrot.iterate(z, z0);
			j++;
			x2 = (int) (OO + z.re()*U);
			y2 = (int) (OO - z.im()*U);

			g.setColor(iterationGray(j, ITERATIONS));
			g.drawLine(x1, y1, x2, y2);
			g.setColor(Color.BLACK);
			x1 = x2;
			y1 = y2;
		}
	}
	
	public void executeDots(Graphics g) {
		Complex z0 = model.getComplex();
		Complex z = z0;
		int j = 0;
		int x, y;

		// Iterative step
		while (z.abs() <= 2 && j <= ITERATIONS) {
			z = Mandelbrot.iterate(z, z0);
			j++;
			x = (int) (OO + z.re()*U);
			y = (int) (OO - z.im()*U);

			g.setColor(iterationBlue(j, ITERATIONS));
			g.drawLine(x - 2, y, x + 2, y);
			g.drawLine(x, y - 2, x, y + 2);
			g.setColor(Color.BLACK);
		}
		// Base step (on top)
		x = (int) (OO + z0.re()*U);
		y = (int) (OO - z0.im()*U);
		g.setColor(Color.MAGENTA);
		g.fillOval(x-4, y-4, 8, 8);
		g.setColor(Color.BLACK);
		System.out.println(j + ":\t" + z);
		
		this.model.setResult((j > ITERATIONS?-1:j));
	}
	
	private static Color iterationGray(int j, int iterations) {
		float pct = (float) j/(iterations + 1);
		int gray = 255 - (int)(pct*128);
		return new Color(gray, gray, gray);
	}
	
	private static Color iterationBlue(int j, int iterations) {
		float pct = (float) j/(iterations + 1);
		int blueInk = (int)(pct*200);
		System.out.println(pct);
		return new Color(200 - blueInk, 200 - blueInk, 255);
	}
}
