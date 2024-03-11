package complex;

/**The Complex class represents a Complex Number
 * specified by a real and and imaginary part.
 * @author Mulliganaceous
 */
public class Complex extends C {
	private double x, y;
	
	public Complex(double real_part, double imaginary_part) {
		this.x = real_part;
		this.y = imaginary_part;
	}
	
	/* COMPLEX OPERATIONS */
	
	public double re() {
		return this.x;
	}
	
	public double im() {
		return this.y;
	}
	
	public boolean isReal() {
		return this.y == 0 || Math.abs(this.arg()) == Math.PI/2;
	}
	
	public double abs() {
		return Math.sqrt(this.x*this.x + this.y*this.y);
	}
	
	public Complex conj() {
		return new Complex(this.x, -this.y);
	}
	
	public double arg() {
		return Math.atan2(this.y, this.x);
	}
	
	/* ARITHMETIC OPERATIONS */
	public Complex add(C other) {
		other = other.rect();
		return new Complex(this.x + other.re(), this.y + other.im());
	}
	
	public Complex sub(C other) {
		other = other.rect();
		return new Complex(this.x - other.re(), this.y - other.im());
	}
	
	public Complex mult(C other) {
		other = other.rect();
		double re = this.re()*other.re() - this.im()*other.im();
		double im = this.re()*other.im() + this.im()*other.re();
		return new Complex(re, im);
	}
	
	public Complex div(C other) {
		other = other.rect();
		double denom = other.abs()*other.abs();
		double reNum = this.re()*other.re() + this.im()*other.im();
		double imNum = -this.re()*other.im() + this.im()*other.re();
		return new Complex(reNum/denom, imNum/denom);
	}
	
	public Complex raise(C exponent) { // TODO
		exponent = exponent.rect();
		if (this.im() == 0) {
			double log = Math.log(this.abs());
			double reExp = exponent.re()*log;
			double imExp = exponent.im()*log;
			double re = Math.exp(reExp)*(Math.cos(imExp));
			double im = Math.exp(reExp)*(Math.sin(imExp));
			return new Complex(re, im);
		}
		return new Complex(-1, 0);
	}
	
	/* CONVERSION */
	public C convert() {
		// TODO Auto-generated method stub
		return null;
	}

	public Complex rect() {
		return this;
	}

	public ComplexPolar polar() {
		// TODO Auto-generated method stub
		return null;
	}
	
	public String toString() {
		String sgn = this.y >= 0 ? "+" : "-";
		return "[" + this.x + " " + sgn + " i" +  Math.abs(this.y) + "]";
	}
	
	public String toStringDigits(int k) {
		String sgn = this.y >= 0 ? "+" : "-";
		String digitsX = String.format("%." + k + "f", this.x);
		String digitsY = String.format("%." + k + "f", this.y);
		return "[" + digitsX + " " + sgn + " i" +  digitsY + "]";
	}
}
