package guiDemo;

import javax.swing.JFrame;

import complex.Complex;

public class MandelbrotTraceWindow extends JFrame {
	private static final long serialVersionUID = -1;
	final int WINDOW_WIDTH = 800;
	final int WINDOW_HEIGHT = 600;
	
	public MandelbrotTraceWindow() {
		super();
		this.setLayout(null);
		
		MandelbrotTraceModel model = new MandelbrotTraceModel();
		Complex z0 = demoComplex();
		model.setComplex(z0);
		MandelbrotTracePanel mPanel = new MandelbrotTracePanel(model);
		this.getContentPane().add(mPanel);
		
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.pack();
		this.setSize(800,600);
		this.setResizable(false);
		this.setTitle("Mandelbrot Trace Demo");
		this.setVisible(true);
	}
	
	static Complex demoComplex() {
		final double re = -0.706656415693871;
		final double im = +0.236382660750009;
		return new Complex(re, im);
	}
	
	public static void main(String[] args) {
		MandelbrotTraceWindow window = new MandelbrotTraceWindow();
	}
}
