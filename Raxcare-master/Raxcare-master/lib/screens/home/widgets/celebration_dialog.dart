import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:confetti/confetti.dart';
import 'dart:math' show pi;

class CelebrationDialog extends StatelessWidget {
  final ConfettiController confettiController;
  final int daysSober;

  const CelebrationDialog({
    Key? key,
    required this.confettiController,
    required this.daysSober,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      backgroundColor: Colors.white.withOpacity(0.9),
      content: Stack(
        clipBehavior: Clip.none,
        children: [
          ConfettiWidget(
            confettiController: confettiController,
            blastDirection: -pi / 2,
            maxBlastForce: 5,
            minBlastForce: 2,
            emissionFrequency: 0.05,
            numberOfParticles: 50,
            gravity: 0.1,
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.stars, size: 60, color: Colors.amber),
                const SizedBox(height: 16),
                Text(
                  'Amazing Achievement!',
                  style: GoogleFonts.poppins(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue[900],
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  '$daysSober Days Sober!',
                  style: GoogleFonts.poppins(
                    fontSize: 20,
                    color: Colors.blue[700],
                  ),
                ),
                const SizedBox(height: 16),
                Text(
                  'You\'re doing incredible! Keep going strong!',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.poppins(
                    fontSize: 16,
                    color: Colors.grey[700],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: Text(
            'Thank you!',
            style: GoogleFonts.poppins(fontSize: 16),
          ),
        ),
      ],
    );
  }
}
