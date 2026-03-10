package turing;

public class Tape {
	public final char BLANK;
	private Node first; // Index of symbol furthest back
	private Node cur;     // Index of current head
	public Tape(char[] alphabet) {
		this.BLANK = alphabet[0];
		this.first = new Node(0,BLANK);
		this.cur = this.first;
	}
	
	public Tape(char[] alphabet, String input, int offset) {
		this.BLANK = alphabet[0];
		if (input.equals(""))
			input = BLANK + "";
		
		this.first = new Node(0,BLANK);
		this.cur = this.first;
		Node head = this.first;
		
		int x = 0;
		if (offset > x) {
			while (offset > x) {
				this.fwd();
				x++;
			}
		}
		else if (offset < x) {
			while (offset < x) {
				this.rev();
				x--;
			}
		}
		
		this.write(input.charAt(0));
		for (int k = 1; k < input.length(); k++) {
			this.fwd();
			this.write(input.charAt(k));
		}
		this.cur = head;
	}
	
	public char curSymbol() {
		return this.cur.s;
	}
	
	public void fwd() {
		if (this.cur.next == null) {
			this.cur.next = new Node(this.cur.x + 1, BLANK);
			this.cur.next.prev = this.cur;
		}
		this.cur = this.cur.next;
	}
	
	public void rev() {
		if (this.cur == this.first) {
			this.first.prev = new Node(this.cur.x - 1, BLANK);
			this.first.prev.next = this.cur;
			this.first = this.cur.prev;
		}
		this.cur = this.cur.prev;
	}
	
	public void write(char s) {
		this.cur.s = s;
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		Node n = this.first;
		while (n != null) {
			if (n.x == 0)
				sb.append("|");
			if (n == this.cur)
				sb.append("{" + n.toString() + "}");
			else
				sb.append(" " + n.toString() + " ");
			n = n.next;
		}
		return sb.toString();
	}
	
	private class Node {
		private final int x;
		private char s;
		private Node prev, next;
		public Node(int x, char s) {
			this.x = x;
			this.s = s;
			this.prev = null;
			this.next = null;
		}
		
		public String toString() {
			return this.s + "";
		}
	}
}
