package complex;

public class ComplexPolar extends C {
	int abs, arg;
	
	public ComplexPolar(int magnitude, int angle) {
		this.abs = magnitude;
		this.arg = angle;
	}
	
	@Override
	public double re() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double im() {
		// TODO Auto-generated method stub
		return 0;
	}
	
	@Override
	public double sqnorm() {
		return 0;
	}

	@Override
	public double abs() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double arg() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public C conj() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public C add(C other) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public C sub(C other) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public C mult(C other) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public C div(C other) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public C convert() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Complex rect() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ComplexPolar polar() {
		// TODO Auto-generated method stub
		return null;
	}

}
