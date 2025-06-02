// ignore_for_file: prefer_const_constructors, prefer_const_literals_to_create_immutables, non_constant_identifier_names
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:lottie/lottie.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:glassmorphism/glassmorphism.dart';
import 'package:shimmer/shimmer.dart';
import 'package:url_launcher/url_launcher.dart';

class About extends StatefulWidget {
  const About({Key? key}) : super(key: key);

  @override
  _AboutState createState() => _AboutState();
}

class _AboutState extends State<About> with SingleTickerProviderStateMixin {
  late TabController _tabController;
  final ScrollController _scrollController = ScrollController();
  bool _isExpanded = false;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final textTheme = theme.textTheme;

    return Scaffold(
      body: NestedScrollView(
        controller: _scrollController,
        headerSliverBuilder: (context, innerBoxIsScrolled) {
          return [
            SliverAppBar(
              expandedHeight: 300,
              floating: false,
              pinned: true,
              elevation: 0,
              backgroundColor: Colors.transparent,
              flexibleSpace: FlexibleSpaceBar(
                background: Stack(
                  fit: StackFit.expand,
                  children: [
                    // Parallax background
                    Image.asset(
                      'assets/recovery_bg.jpg',
                      fit: BoxFit.cover,
                    ),
                    // Gradient overlay
                    Container(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [
                            theme.primaryColor.withOpacity(0.7),
                            theme.primaryColor.withOpacity(0.9),
                          ],
                        ),
                      ),
                    ),
                    // Content
                    SafeArea(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          // Logo with animation
                          GlassmorphicContainer(
                            width: 120,
                            height: 120,
                            borderRadius: 60,
                            blur: 10,
                            alignment: Alignment.center,
                            border: 1,
                            linearGradient: LinearGradient(
                              colors: [
                                Colors.white.withOpacity(0.2),
                                Colors.white.withOpacity(0.1),
                              ],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            borderGradient: LinearGradient(
                              colors: [
                                Colors.white.withOpacity(0.5),
                                Colors.white.withOpacity(0.2),
                              ],
                            ),
                            child: Lottie.asset(
                              'assets/recovery_animation.json',
                              height: 80,
                              width: 80,
                              fit: BoxFit.contain,
                            ),
                          ).animate().fadeIn(duration: 800.ms).scale(),
                          SizedBox(height: 16),
                          // App name with shimmer effect
                          Shimmer.fromColors(
                            baseColor: Colors.white,
                            highlightColor: Colors.white.withOpacity(0.5),
                            period: Duration(seconds: 3),
                            child: Text(
                              'RaxCare',
                              style: GoogleFonts.montserrat(
                                fontSize: 38,
                                fontWeight: FontWeight.w900,
                                letterSpacing: 1.2,
                              ),
                            ),
                          ),
                          SizedBox(height: 8),
                          // Tagline
                          Text(
                            'Redefining Recovery with AI & CBT',
                            style: GoogleFonts.poppins(
                              fontSize: 16,
                              fontWeight: FontWeight.w500,
                              color: Colors.white.withOpacity(0.9),
                              letterSpacing: 0.5,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              bottom: TabBar(
                controller: _tabController,
                indicatorColor: theme.colorScheme.secondary,
                indicatorWeight: 3,
                labelColor: Colors.white,
                unselectedLabelColor: Colors.white.withOpacity(0.7),
                tabs: [
                  Tab(text: 'ABOUT'),
                  Tab(text: 'FEATURES'),
                  Tab(text: 'OUR VISION'),
                ],
              ),
            ),
          ];
        },
        body: TabBarView(
          controller: _tabController,
          children: [
            // ABOUT TAB
            _buildAboutTab(theme),

            // FEATURES TAB
            _buildFeaturesTab(theme),

            // VISION TAB
            _buildVisionTab(theme),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () {
          // Action for getting started
        },
        icon: Icon(Icons.play_arrow_rounded),
        label: Text('Start Recovery Journey'),
        backgroundColor: theme.colorScheme.secondary,
      ),
    );
  }

  Widget _buildAboutTab(ThemeData theme) {
    return SingleChildScrollView(
      padding: EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Mission Statement
          _buildSectionTitle('Our Mission', theme),
          SizedBox(height: 16),
          Text(
            "RaxCare represents a revolutionary approach to addiction recovery, combining cutting-edge artificial intelligence with evidence-based Cognitive Behavioral Therapy (CBT) techniques. We've created a comprehensive ecosystem that supports individuals throughout their entire recovery journey — from acknowledging the addiction to maintaining long-term sobriety and preventing relapse.",
            style: GoogleFonts.roboto(
              fontSize: 16,
              height: 1.6,
              color: theme.textTheme.bodyLarge?.color,
            ),
          ),
          SizedBox(height: 16),
          Text(
            "Unlike traditional recovery programs that often rely solely on willpower or external support, RaxCare provides a personalized, accessible recovery companion that adapts to your unique challenges, triggers, and progress patterns. Our AI engine continuously learns from your interactions, becoming increasingly attuned to your specific recovery needs.",
            style: GoogleFonts.roboto(
              fontSize: 16,
              height: 1.6,
              color: theme.textTheme.bodyLarge?.color,
            ),
          ),

          SizedBox(height: 32),

          // The Science Behind
          _buildSectionTitle('The Science', theme),
          SizedBox(height: 16),

          Card(
            elevation: 4,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: Padding(
              padding: EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.psychology,
                        color: theme.primaryColor,
                        size: 28,
                      ),
                      SizedBox(width: 12),
                      Text(
                        'Evidence-Based Approach',
                        style: GoogleFonts.poppins(
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                          color: theme.primaryColor,
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 16),
                  Text(
                    "RaxCare's methodology is grounded in decades of addiction research and cognitive science. Our platform integrates principles from:",
                    style: GoogleFonts.roboto(
                      fontSize: 16,
                      height: 1.5,
                    ),
                  ),
                  SizedBox(height: 16),
                  _buildBulletPoint("Cognitive Behavioral Therapy (CBT): Restructuring negative thought patterns and developing healthier coping mechanisms", theme),
                  _buildBulletPoint("Motivational Enhancement Therapy (MET): Building and sustaining motivation for change", theme),
                  _buildBulletPoint("Dialectical Behavior Therapy (DBT): Developing mindfulness and emotional regulation skills", theme),
                  _buildBulletPoint("Relapse Prevention Models: Identifying triggers and implementing prevention strategies", theme),
                ],
              ),
            ),
          ),

          SizedBox(height: 32),

          // Our Story
          _buildSectionTitle('Our Story', theme),
          SizedBox(height: 16),
          Text(
            "RaxCare was born from a powerful combination of personal experience and technological innovation. Our founding team includes individuals who have navigated their own recovery journeys, alongside AI specialists and mental health professionals who recognized the potential for technology to transform addiction treatment.",
            style: GoogleFonts.roboto(
              fontSize: 16,
              height: 1.6,
              color: theme.textTheme.bodyLarge?.color,
            ),
          ),
          SizedBox(height: 16),
          Text(
            "We recognized that traditional recovery methods, while valuable, often fail to provide the consistent, personalized support many need to succeed long-term. RaxCare fills this gap by offering immediate, judgment-free guidance exactly when users need it most—whether that's during a 3 AM craving or while navigating a high-risk social situation.",
            style: GoogleFonts.roboto(
              fontSize: 16,
              height: 1.6,
              color: theme.textTheme.bodyLarge?.color,
            ),
          ),
          SizedBox(height: 16),
          Text(
            "The name 'RaxCare' combines 'Relax' and 'Care'—embodying our philosophy that recovery should be a supportive, compassionate process that helps individuals find calm amid chaos while providing comprehensive care for their wellbeing.",
            style: GoogleFonts.roboto(
              fontSize: 16,
              height: 1.6,
              color: theme.textTheme.bodyLarge?.color,
            ),
          ),

          SizedBox(height: 32),
        ],
      ),
    );
  }

  Widget _buildFeaturesTab(ThemeData theme) {
    return SingleChildScrollView(
      padding: EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionTitle('Core Features', theme),
          SizedBox(height: 24),

          // AI Companion
          _buildFeatureCard(
            context: context,
            icon: FontAwesomeIcons.robot,
            title: 'AI Recovery Companion',
            description: 'Our advanced AI uses natural language processing and machine learning to provide personalized guidance, coping strategies, and motivational support tailored to your specific addiction patterns and recovery progress.',
            color: Colors.blue,
          ),

          // CBT Toolkit
          _buildFeatureCard(
            context: context,
            icon: FontAwesomeIcons.brain,
            title: 'Comprehensive CBT Toolkit',
            description: 'Access over 200+ interactive exercises designed by clinical psychologists to address negative thought patterns, emotional triggers, and develop healthier coping mechanisms. Each exercise adapts based on your responses and progress.',
            color: Colors.purple,
          ),

          // Progress Analytics
          _buildFeatureCard(
            context: context,
            icon: FontAwesomeIcons.chartLine,
            title: 'Advanced Progress Analytics',
            description: 'Track your recovery journey with detailed metrics and visualizations. Monitor sobriety streaks, identify pattern correlations between moods and cravings, and celebrate milestones with achievement badges that reinforce positive behavior.',
            color: Colors.green,
          ),

          // Trigger Management
          _buildFeatureCard(
            context: context,
            icon: FontAwesomeIcons.triangleExclamation,
            title: 'Personalized Trigger Management',
            description: 'Our AI helps identify your unique addiction triggers through behavioral pattern recognition. Develop customized prevention strategies for high-risk situations and receive real-time interventions when the app detects potential relapse patterns.',
            color: Colors.orange,
          ),

          // Journal
          _buildFeatureCard(
            context: context,
            icon: FontAwesomeIcons.bookOpen,
            title: 'Therapeutic Journal & Mood Tracking',
            description: 'Document your thoughts, emotions, and experiences in a secure digital journal. Our sentiment analysis helps identify emotional patterns and provides insights into your recovery progress. Track daily mood fluctuations to better understand your emotional triggers.',
            color: Colors.teal,
          ),

          // Community
          _buildFeatureCard(
            context: context,
            icon: FontAwesomeIcons.userGroup,
            title: 'Anonymous Community Support',
            description: 'Connect with others on similar recovery journeys through moderated, anonymous group discussions. Share experiences, strategies, and encouragement within a safe, judgment-free environment vetted by our AI moderation system.',
            color: Colors.indigo,
          ),

          SizedBox(height: 32),

          _buildSectionTitle('Technical Innovation', theme),
          SizedBox(height: 16),

          Container(
            padding: EdgeInsets.all(24),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  theme.colorScheme.secondary.withOpacity(0.1),
                  theme.primaryColor.withOpacity(0.1),
                ],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: theme.primaryColor.withOpacity(0.2),
                width: 1,
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "RaxCare leverages cutting-edge technology to create an unprecedented recovery experience:",
                  style: GoogleFonts.roboto(
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                    height: 1.5,
                  ),
                ),
                SizedBox(height: 16),
                _buildTechFeature(
                  "Natural Language Processing",
                  "Our AI understands the nuances of your communication, detecting emotional undertones and potential risk factors even when they're not explicitly stated.",
                  theme,
                ),
                _buildTechFeature(
                  "Behavioral Pattern Recognition",
                  "The app learns your unique usage patterns, sleep cycles, and activity levels to predict vulnerable moments before they occur.",
                  theme,
                ),
                _buildTechFeature(
                  "Secure Encryption",
                  "Military-grade encryption ensures your personal data and journal entries remain completely private and secure.",
                  theme,
                ),
                _buildTechFeature(
                  "Offline Functionality",
                  "Critical features remain accessible even without internet connection, ensuring support is available whenever you need it.",
                  theme,
                ),
              ],
            ),
          ),

          SizedBox(height: 32),
        ],
      ),
    );
  }

  Widget _buildVisionTab(ThemeData theme) {
    return SingleChildScrollView(
      padding: EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildSectionTitle('Our Philosophy', theme),
          SizedBox(height: 16),

          Text(
            "At RaxCare, we believe that successful recovery requires more than just abstinence—it demands a holistic transformation of thought patterns, emotional responses, and lifestyle choices. Our approach is guided by four core principles:",
            style: GoogleFonts.roboto(
              fontSize: 16,
              height: 1.6,
              color: theme.textTheme.bodyLarge?.color,
            ),
          ),
          SizedBox(height: 24),

          // Principles Cards
          _buildPrincipleCard(
            context: context,
            number: "01",
            title: "Personalization Above All",
            description: "No two addiction journeys are identical. RaxCare adapts to your specific struggles, strengths, and circumstances rather than imposing a one-size-fits-all approach.",
            theme: theme,
          ),

          _buildPrincipleCard(
            context: context,
            number: "02",
            title: "Accessibility & Consistency",
            description: "Recovery support should be available exactly when you need it—not just during scheduled sessions. Our AI companion provides immediate, consistent guidance at any hour, anywhere in the world.",
            theme: theme,
          ),

          _buildPrincipleCard(
            context: context,
            number: "03",
            title: "Empowerment Through Insight",
            description: "Understanding the 'why' behind your addiction is crucial for lasting change. RaxCare helps you recognize patterns, triggers, and underlying factors driving addictive behaviors.",
            theme: theme,
          ),

          _buildPrincipleCard(
            context: context,
            number: "04",
            title: "Sustainable Integration",
            description: "Recovery isn't separate from life—it's integrated within it. Our approach helps you build recovery practices that complement rather than disrupt your daily existence.",
            theme: theme,
          ),

          SizedBox(height: 32),

          _buildSectionTitle('Looking Forward', theme),
          SizedBox(height: 16),

          Container(
            padding: EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: theme.cardColor,
              borderRadius: BorderRadius.circular(16),
              boxShadow: [
                BoxShadow(
                  color: Colors.black12,
                  blurRadius: 10,
                  offset: Offset(0, 4),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Our vision extends beyond current capabilities. The RaxCare roadmap includes:",
                  style: GoogleFonts.roboto(
                    fontSize: 16,
                    height: 1.6,
                  ),
                ),
                SizedBox(height: 16),

                _buildRoadmapItem(
                  "Biometric Integration",
                  "Syncing with wearable devices to detect physiological signs of stress and craving before you're consciously aware of them.",
                  theme,
                ),

                _buildRoadmapItem(
                  "Predictive Crisis Prevention",
                  "Using advanced data analytics to forecast potential relapse scenarios days before they might occur.",
                  theme,
                ),

                _buildRoadmapItem(
                  "Personalized Neuroplasticity Exercises",
                  "Custom brain training activities designed to strengthen the neural pathways associated with healthier choices and responses.",
                  theme,
                ),

                _buildRoadmapItem(
                  "Post-Recovery Life Planning",
                  "Tools to help rebuild life structures, relationships, and purpose beyond the initial recovery phase.",
                  theme,
                ),

                SizedBox(height: 16),
                Text(
                  "Our ultimate goal is to make high-quality, evidence-based addiction recovery support accessible to everyone who needs it, regardless of geographic location, financial resources, or severity of addiction. We envision a world where recovery is not just possible but sustainable for every individual seeking change.",
                  style: GoogleFonts.roboto(
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                    fontStyle: FontStyle.italic,
                    height: 1.6,
                    color: theme.primaryColor,
                  ),
                ),
              ],
            ),
          ),

          SizedBox(height: 40),

          // Team quote
          Center(
            child: Container(
              width: double.infinity,
              padding: EdgeInsets.symmetric(vertical: 24),
              child: Column(
                children: [
                  Icon(
                    Icons.format_quote,
                    size: 40,
                    color: theme.colorScheme.secondary,
                  ),
                  SizedBox(height: 16),
                  Text(
                    "Recovery isn't just about stopping something negative—it's about starting something positive. At RaxCare, we're committed to helping you build a life so fulfilling that there's no room for addiction.",
                    textAlign: TextAlign.center,
                    style: GoogleFonts.playfairDisplay(
                      fontSize: 20,
                      fontWeight: FontWeight.w500,
                      fontStyle: FontStyle.italic,
                      height: 1.6,
                      color: theme.colorScheme.secondary,
                    ),
                  ),
                  SizedBox(height: 16),
                  Text(
                    "— The RaxCare Team",
                    style: GoogleFonts.poppins(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ),

          SizedBox(height: 32),
        ],
      ),
    );
  }

  // Helper Methods
  Widget _buildSectionTitle(String title, ThemeData theme) {
    return Row(
      children: [
        Container(
          height: 24,
          width: 4,
          decoration: BoxDecoration(
            color: theme.colorScheme.secondary,
            borderRadius: BorderRadius.circular(4),
          ),
        ),
        SizedBox(width: 12),
        Text(
          title,
          style: GoogleFonts.poppins(
            fontSize: 24,
            fontWeight: FontWeight.bold,
            color: theme.colorScheme.secondary,
          ),
        ),
      ],
    ).animate().fadeIn(duration: 500.ms).moveX(begin: -20, end: 0);
  }

  Widget _buildBulletPoint(String text, ThemeData theme) {
    return Padding(
      padding: EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            Icons.check_circle,
            size: 18,
            color: theme.colorScheme.secondary,
          ),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: GoogleFonts.roboto(
                fontSize: 15,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFeatureCard({
    required BuildContext context,
    required IconData icon,
    required String title,
    required String description,
    required Color color,
  }) {
    final theme = Theme.of(context);

    return Container(
      margin: EdgeInsets.only(bottom: 20),
      child: Card(
        elevation: 4,
        shadowColor: color.withOpacity(0.3),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        child: Padding(
          padding: EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: color.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: FaIcon(
                      icon,
                      color: color,
                      size: 28,
                    ),
                  ),
                  SizedBox(width: 16),
                  Expanded(
                    child: Text(
                      title,
                      style: GoogleFonts.poppins(
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                        color: theme.colorScheme.secondary,
                      ),
                    ),
                  ),
                ],
              ),
              SizedBox(height: 16),
              Text(
                description,
                style: GoogleFonts.roboto(
                  fontSize: 15,
                  height: 1.6,
                ),
              ),
              SizedBox(height: 8),
              TextButton(
                onPressed: () {
                  // Feature detail action
                },
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      'Learn more',
                      style: TextStyle(
                        color: color,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    SizedBox(width: 4),
                    Icon(
                      Icons.arrow_forward,
                      size: 16,
                      color: color,
                    ),
                  ],
                ),
                style: TextButton.styleFrom(
                  padding: EdgeInsets.zero,
                  alignment: Alignment.centerLeft,
                ),
              ),
            ],
          ),
        ),
      ),
    ).animate().fadeIn(duration: 400.ms).moveY(begin: 20, end: 0);
  }

  Widget _buildTechFeature(String title, String description, ThemeData theme) {
    return Padding(
      padding: EdgeInsets.only(bottom: 20),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            margin: EdgeInsets.only(top: 4),
            height: 8,
            width: 8,
            decoration: BoxDecoration(
              color: theme.colorScheme.secondary,
              shape: BoxShape.circle,
            ),
          ),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.poppins(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: theme.colorScheme.secondary,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  description,
                  style: GoogleFonts.roboto(
                    fontSize: 14,
                    height: 1.5,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPrincipleCard({
    required BuildContext context,
    required String number,
    required String title,
    required String description,
    required ThemeData theme,
  }) {
    return Container(
      margin: EdgeInsets.only(bottom: 24),
      padding: EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: theme.cardColor,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            number,
            style: GoogleFonts.montserrat(
              fontSize: 32,
              fontWeight: FontWeight.bold,
              color: theme.primaryColor.withOpacity(0.2),
            ),
          ),
          SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.poppins(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: theme.colorScheme.secondary,
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  description,
                  style: GoogleFonts.roboto(
                    fontSize: 15,
                    height: 1.6,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    ).animate().fadeIn(duration: 400.ms, delay: (int.parse(number) * 100).ms);
  }

  Widget _buildRoadmapItem(String title, String description, ThemeData theme) {
    return Padding(
      padding: EdgeInsets.only(bottom: 16),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            Icons.arrow_right,
            size: 24,
            color: theme.primaryColor,
          ),
          SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.poppins(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: theme.primaryColor,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  description,
                  style: GoogleFonts.roboto(
                    fontSize: 14,
                    height: 1.5,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}