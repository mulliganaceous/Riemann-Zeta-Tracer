package guiDemo;

import javax.swing.JFrame;

import complex.Complex;

public class RiemannZetaWindow extends JFrame {
	private static final long serialVersionUID = -1;
	private static final int WINDOW_WIDTH = 1000;
	private static final int WINDOW_HEIGHT = 648;
	
	public RiemannZetaWindow() {
		super();
		this.setLayout(null);
		
		RiemannZetaCriticalModel model = new RiemannZetaCriticalModel();
		RiemannZetaPanel mPanel = new RiemannZetaPanel(model);
		RiemannZetaSelection sPanel = new RiemannZetaSelection(model);
		this.getContentPane().add(mPanel);
		this.getContentPane().add(sPanel);
		
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.pack();
		this.setSize(WINDOW_WIDTH,WINDOW_HEIGHT);
		this.setResizable(false);
		this.setTitle("Critical Line Tracer Demo");
		this.setVisible(true);
	}
	
	public static void main(String[] args) {
		RiemannZetaWindow window = new RiemannZetaWindow();
	}
}
