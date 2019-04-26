#!/usr/bin/perl
use strict;
my @data;
my %missingF=();
my %missingS=();
my $count=0;
my $ok=0;
my $hs=0;
while (<STDIN>) {
	chomp($_);
	$_ =~ s/^\s+|\s+$//g; 
	my ($f,$s)=split/\s+/,$_;
	if($f==$s) {
		$ok++;
	} else {
		$missingF{$f}=$count;
		$missingS{$s}=$count;
	}
	$count++;
}
foreach (keys(%missingF)) {
	$hs+=(1-(abs($missingS{$_}-$missingF{$_})/$count)) if(exists($missingS{$_}));
}
$count=1 if($count==0);
my $score=($ok+$hs)/$count; 
print "Score=$score\n";
