package org.missdc.utils.search.tree;

import org.missdc.utils.bitset.IBitSet;

import java.util.function.Consumer;

/**
 * The <code>LeafNode</code> contains exactly one bit set
 */
public class LeafNode implements Node {

	protected final IBitSet set;

	public LeafNode(IBitSet set) {
		this.set = set;
	}

	public Node add(IBitSet set, int bit) {
		if (this.set.equals(set)) {
			return null;
		}
		return InterNode.create(this, new LeafNode(set), bit);
	}

	public Node remove(IBitSet set) {
		return set.equals(this.set) ? EMPTY : null;
	}

	public IBitSet findSubSet(IBitSet of) {
		return set.isSubSetOf(of) ? set : null;
	}

	public IBitSet findSubSet(IBitSet of, IBitSet after) {
		return set.isSubSetOf(of) && set.compareTo(after) > 0 ? set : null;
	}

	public IBitSet findSuperSet(IBitSet of) {
		return of.isSubSetOf(set) ? set : null;
	}

	public IBitSet findSuperSet(IBitSet of, IBitSet after) {
		return of.isSubSetOf(set) && set.compareTo(after) > 0 ? set : null;
	}

	public IBitSet union() {
		return set;
	}

	public IBitSet inter() {
		return set;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder();
		sb.append(getClass().getSimpleName()).append('{');
		sb.append(set);
		return sb.append('}').toString();
	}

	@Override
	public IBitSet findSuperSet(IBitSet of, int without) {
		return !set.get(without) && of.isSubSetOf(set) ? set : null;
	}

	@Override
	public void each(Consumer<IBitSet> consumer) {
		consumer.accept(set);
	}

	@Override
	public void eachSubSet(IBitSet of, Consumer<IBitSet> consumer) {
		if (set.isSubSetOf(of))
			consumer.accept(set);
	}

	@Override
	public void eachSuperSet(IBitSet of, Consumer<IBitSet> consumer) {
		if (of.isSubSetOf(set))
			consumer.accept(set);
	}
}