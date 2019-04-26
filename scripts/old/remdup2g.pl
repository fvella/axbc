#!/usr/bin/perl
use strict;
my @data=();
my %graph;
my %tdv;
my %udv;
my @tmpl;
@data=<STDIN>;
foreach (@data) {
	chomp($_);
	my ($v1, $v2)=split/\t/,$_;
	push (@{$graph{$v1}},$v2);
}
foreach (keys %graph) {
	if(@{ $graph{$_} }==2) {
#		warn "$_: @{ $graph{$_} }\n";	
		push(@{$tdv{$_}},@{ $graph{$_} });
	} 
}
foreach (keys %tdv) {
	@tmpl=();
        foreach (@{$tdv{$_}}) {
		if(!exists($udv{$_})) {
			$udv{$_}=1;
			push @tmpl,$_;
		}
	}	
	@{$graph{$_}}=();
	if(@tmpl>0) {
		push (@{$graph{$_}},@tmpl);
#		warn "$_: @tmpl\n";	
	}
}
foreach (sort { $a <=> $b} (keys %graph)) {
	my $k=$_;
	if(@{ $graph{$_} }>0) {
		foreach(@{ $graph{$_} }) {
			print "$k\t$_\n";
		}
	}
}
