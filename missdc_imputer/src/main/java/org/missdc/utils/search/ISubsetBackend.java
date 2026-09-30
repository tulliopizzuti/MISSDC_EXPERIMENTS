package org.missdc.utils.search;


import org.missdc.utils.bitset.IBitSet;

import java.util.Set;
import java.util.function.Consumer;

public interface ISubsetBackend {

	boolean add(IBitSet bs);

	Set<IBitSet> getAndRemoveGeneralizations(IBitSet invalidFD);

	boolean containsSubset(IBitSet add);

	void forEach(Consumer<IBitSet> consumer);

}