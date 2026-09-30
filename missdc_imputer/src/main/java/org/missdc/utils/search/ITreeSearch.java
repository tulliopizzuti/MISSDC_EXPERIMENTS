package org.missdc.utils.search;


import org.missdc.utils.bitset.IBitSet;
import java.util.Collection;
import java.util.function.Consumer;

public interface ITreeSearch {

	boolean add(IBitSet bs);

	void forEachSuperSet(IBitSet bitset, Consumer<IBitSet> consumer);

	void forEach(Consumer<IBitSet> consumer);

	void remove(IBitSet remove);

	boolean containsSubset(IBitSet bitset);

	Collection<IBitSet> getAndRemoveGeneralizations(IBitSet invalidDC);

}