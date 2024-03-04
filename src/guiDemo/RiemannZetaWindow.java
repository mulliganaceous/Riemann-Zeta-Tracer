package guiDemo;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GraphicsEnvironment;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.Toolkit;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.WindowConstants;

import complex.Complex;

/**Window for the Riemann Zeta simulator
 * @author Mulliganaceous
 */
public class RiemannZetaWindow extends JFrame {
	private static final long serialVersionUID = -1;
	private static final int WINDOW_WIDTH = 640;
	private static final int WINDOW_HEIGHT = 480;
	private static final int SLEEPDUR = 250;
	RiemannZetaCriticalModel model = new RiemannZetaCriticalModel();
	RiemannZetaPanel mPanel = new RiemannZetaPanel(model);
	RiemannZetaSelection sPanel = new RiemannZetaSelection(model);
	
	public RiemannZetaWindow() {
		super();
		this.setLayout(new BorderLayout());
		this.getContentPane().add(mPanel);
		// this.getContentPane().add(sPanel);
		
		// Set visible
		this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		this.pack();
		this.setSize(new Dimension(WINDOW_WIDTH + this.getInsets().left + this.getInsets().right,WINDOW_HEIGHT + this.getInsets().top + this.getInsets().bottom));
		Image icon;
		try {
			icon = ImageIO.read(new File("images/Riemann65536.gif"));
			this.setIconImage(icon);
		} catch (IOException e) {
			e.printStackTrace();
		};
		this.setTitle("Million Dollar Limaçon");
		this.setResizable(false);
		this.setVisible(true);
	}
	
	void loop() {
		model.setInterval(1);
		model.setS(new Complex(0.5, 0));
		while(true) {
			try {
				Thread.sleep(SLEEPDUR);
				Complex s = model.getS();
				Complex z = model.getZetaS();
				System.out.printf("ζ(%.2f + %.6f\u2148) =\t %.6f %c %.6f\u2148\t(|ζ| = %.6f)\n", 
						s.re(), s.im(), z.re(), z.im() >= 0 ? '+' : '-', Math.abs(z.im()), z.abs());
				model.increment();
				mPanel.repaint();
			} catch (Exception exception) {
				exception.printStackTrace();
			}
		}
	}
	
	public static void main(String[] args) {
		RiemannZetaWindow window = new RiemannZetaWindow();
		window.loop();
	}
}
