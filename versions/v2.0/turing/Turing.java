package turing;

import java.util.Arrays;
import java.util.HashMap;

import turing.TuringTransition.Transition;

public class Turing {
	// Fixed attributes of a Turing machine
	private final int Q;
	private final char[] S;
	private final TuringTransition delta;
	// Current status of a Turing machine
	private Tape t;
	private int q;
	private byte runningStatus;
	public final static byte RUNNING = 2;
	public final static byte APPROVED = 1;
	public final static byte REJECTED = 0;
	public final static byte FORWARD = 1;
	public final static byte BACKWARD = -1;
	private HashMap<Character, Integer> charMap;
	
	/**Construct a Turing machine with numerically labelled states,
	 * the starting state being number 0, and an alphabet set with
	 * the first letter being the blank symbol.
	 * @param states
	 * @param alphabet
	 */
	public Turing(int states, char[] alphabet) {
		this.Q = states;
		this.q = 0;
		this.S = alphabet;
		this.delta = new TuringTransition(this, states, alphabet);
		this.t = new Tape(this.S);
		this.runningStatus = RUNNING;
		this.charMap = new HashMap<Character, Integer>();
		for (int k = 0; k < alphabet.length; k++) {
			this.charMap.put(alphabet[k], k);
		}
	}
	
	public int getStates() {
		return this.Q;
	}
	
	public char[] getAlphabet() {
		return this.S;
	}
	
	public HashMap<Character, Integer> getCharMap() {
		return this.charMap;
	}
	
	public void addTransition(int state, char symbol, int nextState, char nextSymbol, byte dir) {
		this.delta.addTransition(state, this.charMap.get(symbol), nextState, this.charMap.get(nextSymbol), dir);
	}
	
	public void addRejection(int state, char symbol) {
		this.delta.addRejection(state, this.charMap.get(symbol));
	}
	
	public void addAcceptance(int state, char symbol) {
		this.delta.addAcceptance(state, this.charMap.get(symbol));
	}
	
	public void initiate(String input, int offset) {
		this.t = new Tape(this.S, input, offset);
		this.q = 0;
		this.runningStatus = RUNNING;
	}
	
	public void step() {
		Transition d = this.delta.getTransition(q, this.charMap.get(this.t.curSymbol()));
		System.out.print("q" + this.q + ":\t" + this.t + "    \t");
		System.out.print(d + "\n");
		if (d.runningStatus == RUNNING) {
			this.t.write(this.S[d.nextSymbolIndex]);
			if (d.dir == FORWARD)
				this.t.fwd();
			if (d.dir == BACKWARD)
				this.t.rev();
			this.q = d.nextState;
		}
		else if (d.runningStatus == APPROVED) {
			this.runningStatus = APPROVED;
		}
		else {
			this.runningStatus = REJECTED;
		}
	}
	
	public void run(int steps) {
		int k = 0;
		while (k <= steps && this.runningStatus == RUNNING) {
			k++;
			this.step();
		}
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.Q + "Q " + Arrays.toString(this.S) + "\n\n");
		sb.append(delta.toString());
		sb.append(this.q + ":\t");
		sb.append(this.t);
		return sb.toString();
	}
}
