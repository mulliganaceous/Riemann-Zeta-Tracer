package guiDemo;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import complex.Complex;

public class RiemannZetaSelection extends JPanel {
	private static final long serialVersionUID = -1;
	private RiemannZetaCriticalModel model;
	private static JButton graphButton, runButton;
	private static JLabel imLabel, intervalLabel;
	private static JTextField imField, intervalField;
	private static JLabel accuracyLabel, stepIntervalLabel;
	
	public RiemannZetaSelection(RiemannZetaCriticalModel model) {
		super();
		this.model = model;
		this.setSize(new Dimension(168,600));
		this.setLocation(816,8);
		this.setLayout(new GridLayout(2,1));
		this.setBackground(Color.WHITE);
		this.setBorder(BorderFactory.createLineBorder(Color.BLACK, 1));
		
		JPanel topPanel = new JPanel();
		//topPanel.setLayout();
		imLabel = new JLabel("Initial imaginary component:");
		imField = new JTextField(10);
		topPanel.add(imLabel);
		topPanel.add(imField);
		intervalLabel = new JLabel("Imaginary interval to graph:");
		intervalField = new JTextField(5);
		topPanel.add(intervalLabel);
		topPanel.add(intervalField);
		graphButton = new JButton("Graph!");
		graphButton.addActionListener(new GraphActionListener());
		runButton = new JButton("Run forever!");
		runButton.addActionListener(new RunForeverActionListener());
		topPanel.add(graphButton);
		topPanel.add(runButton);
		
		this.add(topPanel);
		
		JPanel botPanel = new JPanel();
		accuracyLabel = new JLabel("Accuracy:" + RiemannZetaCriticalModel.ACCURACY_LEVEL);
		stepIntervalLabel = new JLabel("Step:" + RiemannZetaCriticalModel.INCREMENT_LEVEL);
		botPanel.add(accuracyLabel);
		botPanel.add(stepIntervalLabel);
		this.add(botPanel);
	}
	
	private void graph(double im, double interval) {
		this.model.setS(new Complex(0.5, im));
		this.model.setInterval(interval);
		this.model.graph();
	}
	
	private class GraphActionListener implements ActionListener {
		@Override
		public void actionPerformed(ActionEvent e) {
			try {
				double im = Double.parseDouble(imField.getText());
				double interval = Double.parseDouble(intervalField.getText());
				System.out.printf("Graphing the zeta path on the critical "
						+ "line\nstarting from s = %f to %f\n",
						im, im + interval);
				RiemannZetaSelection.this.graph(im, interval);
			} catch (NumberFormatException exc) {
				System.err.println("Cannot graph zeta path!");
			}
		}
	}
	
	private class RunForeverActionListener implements ActionListener {
		@Override
		public void actionPerformed(ActionEvent e) {
			//RiemannZetaSelection.this.model.run();
		}
	}
}
