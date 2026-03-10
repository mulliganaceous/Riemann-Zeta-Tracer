package guiDemo;

import java.util.Observable;

import complex.Complex;

public class MandelbrotTraceModel extends Observable {
	Complex z;
	static int iterations;
	OrbitTraceCommand command;
	int result;
	
	public MandelbrotTraceModel() {
		this.z = new Complex(0,0);
		MandelbrotTraceModel.iterations = 65536;
		this.command = new OrbitTraceCommand(this);
	}
	
	public void setComplex(Complex z) {
		this.z = z;
	}
	
	public Complex getComplex() {
		return this.z;
	}
	
	public static int getIterations() {
		return MandelbrotTraceModel.iterations;
	}
	
	public void setIterations(int iterations) {
		MandelbrotTraceModel.iterations = iterations;
	}
	
	public OrbitTraceCommand getCommand() {
		return this.command;
	}
	
	public int getResult() {
		return this.result;
	}
	
	public void setResult(int result) {
		this.result = result;
	}
}
